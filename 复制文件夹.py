from math import e
import os, sys
import shutil
import logging
import platform
from datetime import datetime
import ctypes, elevate

class SafeCopier:
    def __init__(self, src_dir, dst_dir):
        self.src_dir = self._format_path(src_dir)
        self.dst_dir = self._format_path(dst_dir)
        self.copied_files = 0
        self.copied_dirs = 0
        self.total_size = 0
        self.long_paths = []
        self.start_time = datetime.now()
        self.error_log = os.path.join(os.path.dirname(__file__), "copy_errors.log")
        self.current_file = ""
        
        # 日志配置
        logging.basicConfig(
            filename=self.error_log,
            level=logging.ERROR,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def _format_path(self, path):
        """处理Windows长路径格式[6,7](@ref)"""
        if platform.system() == 'Windows':
            path = os.path.abspath(path)
            if not path.startswith('\\\\?\\'):
                return '\\\\?\\' + path.lstrip('\\')
        return os.path.abspath(path)

    def _update_progress(self):
        """实时显示复制进度"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        sys.stdout.write(
            f"\r📄 已复制 {self.copied_files} 文件 | 📁 {self.copied_dirs} 文件夹 | "
            f"🗜 {self.total_size/1024/1024:.2f} MB | ⏱ {elapsed:.1f}s"
            f"📄 当前文件: {self.current_file}... | "
        )
        sys.stdout.flush()

    def copy(self):
        """执行安全复制操作"""
        try:
            # 创建目标目录
            if not os.path.exists(self.dst_dir):
                os.makedirs(self.dst_dir)

            # 遍历源目录[5,6](@ref)
            for root, dirs, files in os.walk(self.src_dir, topdown=True, onerror=self._handle_error):
                # 处理目录
                rel_path = os.path.relpath(root, self.src_dir)
                target_dir = os.path.join(self.dst_dir, rel_path)
                if not os.path.exists(target_dir):
                    os.makedirs(self._format_path(target_dir), exist_ok=True)
                    self.copied_dirs += 1

                # 处理文件
                for file in files:
                    src_file = os.path.join(root, file)
                    self.current_file = src_file
                    self._update_progress()
                    dst_file = os.path.join(target_dir, file)
                    try:
                        self.total_size += os.path.getsize(src_file)
                    except Exception as e:
                        logging.error(f"无法获取文件大小: {src_file} | 错误类型: {type(e).__name__}", exc_info=True)
                    self._copy_file(src_file, dst_file)

            # 输出超长路径
            self._save_long_paths()
            
        except KeyboardInterrupt:
            print("\n\n⚠️ 用户中断操作")

    def _copy_file(self, src, dst):
        """安全复制文件（支持长路径）[6,8](@ref)"""
        try:
            # 检查路径长度
            if len(src) > 255:
                self.long_paths.append(src)
                
            # 执行复制
            shutil.copy2(
                self._format_path(src),
                self._format_path(dst)
            )

            self.copied_files += 1
            
        except Exception as e:
            logging.error(f"无法复制 {src} -> {dst} | 错误类型: {type(e).__name__}", exc_info=True)

    def _handle_error(self, error):
        """处理目录访问错误[6](@ref)"""
        error_path = error.filename if hasattr(error, 'filename') else '未知路径'
        logging.error(f"无法访问目录: {error_path} | 错误类型: {type(error).__name__}")
        self.copied_dirs += 0  # 防止进度中断

    def _save_long_paths(self):
        """保存超长路径记录"""
        output_path = os.path.join(os.path.dirname(__file__), 'long_paths.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.long_paths))

def require_admin():
    """检测并请求管理员权限[1,5](@ref)"""
    if ctypes.windll.shell32.IsUserAnAdmin() == 0:
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, __file__, None, 1)
        sys.exit()

if __name__ == '__main__':
    require_admin()
    elevate.elevate()
    src = input("请输入源目录路径：")
    dst = input("请输入目标目录路径：")
    
    print("\n🔍 开始复制...")
    copier = SafeCopier(src, dst)
    copier.copy()
    
    print("\n\n✅ 操作完成！")
    print(f"📊 总计复制：{copier.copied_files} 文件 | {copier.copied_dirs} 文件夹")
    print(f"⚠️ 超长路径已保存至：{os.path.abspath('long_paths.txt')}")
    print(f"⛔ 错误日志路径：{copier.error_log}")