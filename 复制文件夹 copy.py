import os
import shutil, platform
import time
import sys
from datetime import datetime

class RealTimeCopier:
    def __init__(self, src_dir, dst_dir):
        self.src_dir = self._format_path(src_dir)
        self.dst_dir = self._format_path(dst_dir)
        self.copied_files = 0
        self.copied_dirs = 0
        self.total_size = 0
        self.start_time = datetime.now()
        self.last_update = time.time()
        self.current_speed = 0
        
        # 进度显示参数
        self.refresh_interval = 1  # 每秒刷新
        self.current_file = ""

    def _format_path(self, path):
        """处理Windows长路径"""
        if platform.system() == 'Windows':
            path = os.path.abspath(path)
            if not path.startswith('\\\\?\\'):
                return '\\\\?\\' + path.lstrip('\\')
        return os.path.abspath(path)

    def _update_display(self):
        """实时刷新监控数据"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        sys.stdout.write(
            f"\r📄 当前文件: {self.current_file[:50]}... | "
            f"⚡ 速度: {self.current_speed/1024/1024:.2f} MB/s | "
            f"📝 {self.copied_files} 文件 | 📁 {self.copied_dirs} 文件夹 | "
            f"🗜 {self.total_size/1024/1024:.2f} MB | ⏱ {elapsed:.1f}s"
        )
        sys.stdout.flush()

    def copy(self):
        """执行安全复制操作"""
        try:
            # 创建目标目录
            os.makedirs(self.dst_dir, exist_ok=True)

            # 遍历源目录
            for root, dirs, files in os.walk(self.src_dir, topdown=True, onerror=self._handle_error):
                # 处理目录
                rel_path = os.path.relpath(root, self.src_dir)
                target_dir = os.path.join(self.dst_dir, rel_path)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir, exist_ok=True)
                    self.copied_dirs += 1

                # 处理文件
                for file in files:
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(target_dir, file)
                    self._copy_file_with_progress(src_path, dst_path)

        except KeyboardInterrupt:
            print("\n\n⚠️ 用户中断操作")

    def _copy_file_with_progress(self, src, dst):
        """带实时监控的文件复制"""
        self.current_file = src
        start_time = time.time()
        copied_bytes = 0
        
        try:
            # 分块复制实现速度计算
            with open(src, 'rb') as src_f, open(dst, 'wb') as dst_f:
                while True:
                    # 每秒刷新检测
                    if time.time() - self.last_update >= self.refresh_interval:
                        self._update_display()
                        self.last_update = time.time()

                    # 读取数据块
                    data = src_f.read(1024 * 1024)  # 1MB块
                    if not data:
                        break
                    
                    # 写入并统计
                    dst_f.write(data)
                    copied_bytes += len(data)
                    self.total_size += len(data)
                    
                    # 计算瞬时速度
                    time_diff = time.time() - start_time
                    self.current_speed = copied_bytes / time_diff if time_diff > 0 else 0

            self.copied_files += 1
        except Exception as e:
            print(f"\n⛔ 复制失败: {src} -> {dst} | {str(e)}")

    def _handle_error(self, error):
        """处理目录访问错误"""
        pass  # 可扩展错误日志记录

if __name__ == '__main__':
    src = input("请输入源目录路径：")
    dst = input("请输入目标目录路径：")
    
    print("\n🔍 开始复制...")
    copier = RealTimeCopier(src, dst)
    copier.copy()
    
    print("\n\n✅ 操作完成！")
    print(f"📊 总计复制：{copier.copied_files} 文件 | {copier.copied_dirs} 文件夹")