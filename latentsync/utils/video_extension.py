import math
import os
import subprocess
import tempfile
import time


def get_duration(file_path):
    """获取音频或视频文件的时长（秒）"""
    file_path = os.path.abspath(file_path)
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of',
        'default=noprint_wrappers=1:nokey=1', file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)


def process(*, audio_file, video_file, output_file):
    """根据音频和视频时长处理视频"""
    audio_file = os.path.abspath(audio_file)
    video_file = os.path.abspath(video_file)
    output_file = os.path.abspath(output_file)
    # 获取音频和视频时长
    audio_duration = get_duration(audio_file)
    video_duration = get_duration(video_file)
    duration = math.ceil(audio_duration)  # 向上取整，避免剪切视频后的时长小于音频

    if video_duration > audio_duration:
        print("视频时长大于音频时长，裁剪视频...")
        duration = min(video_duration, duration)  # 避免向上取整后的时长大于视频时长
        cmd = f"ffmpeg -loglevel error -i {video_file} -t {duration} -c:v copy -c:a copy {output_file} -y"
        subprocess.run(cmd, shell=True)
    elif video_duration == audio_duration:
        print("视频时长等于音频时长，无需处理...")
        os.system(f"cp {video_file} {output_file}")
    else:
        print("视频时长小于音频时长，进行循环、反转拼接...")
        # 反转视频并保存为临时文件
        reversed_video = os.path.join(tempfile.gettempdir(),
                                      f'reversed_{int(time.time() * 1000)}_{os.path.basename(video_file)}')
        reverse_cmd = f"ffmpeg -loglevel error -i {video_file} -vf 'reverse' -af 'areverse' {reversed_video} -y"
        subprocess.run(reverse_cmd, shell=True)

        # 计算需要多少个原视频和反转视频片段
        remain_num = 1 if audio_duration % video_duration > 0 else 0
        nums = int(audio_duration // video_duration) + remain_num

        # 拼接命令构建
        concat_cmd_parts = []
        flag = True
        for _ in range(nums):
            current_video = video_file if flag else reversed_video
            concat_cmd_parts.append(f"-i {current_video}")
            flag = not flag

        # 构建过滤器复杂指令
        filter_complex = ""
        for i in range(nums):
            filter_complex += f"[{i}:v][{i}:a]"

        filter_complex += f"concat=n={nums}:v=1:a=1[outv][outa]"

        looped_video = os.path.join(tempfile.gettempdir(),
                                    f'looped_{int(time.time())}_{os.path.basename(video_file)}')
        concat_cmd = (
            f"ffmpeg -loglevel error {' '.join(concat_cmd_parts)} "
            f"-filter_complex \"{filter_complex}\" "
            f"-map [outv] -map [outa] {looped_video}"
        )
        subprocess.run(concat_cmd, shell=True)

        # 裁剪至音频时长
        final_cmd = f"ffmpeg -loglevel error -i {looped_video} -t {duration} -c:v copy -c:a copy {output_file} -y"
        subprocess.run(final_cmd, shell=True)

        # 清理临时文件
        os.remove(reversed_video)
        os.remove(looped_video)


if __name__ == "__main__":
    audio = "../../assets/demo2_audio.wav"
    video = "../../assets/demo1_video.mp4"
    output = "../../assets/output_video.mp4"

    process(audio_file=audio, video_file=video, output_file=output)
