import os
import sys
import logging
import platform
from datetime import datetime

class DirectoryScanner:
    def __init__(self, target_dir):
        self.target_dir = self._format_path(target_dir)
        self.file_count = 0
        self.dir_count = 0
        self.total_size = 0
        self.long_paths = []
        self.access_errors = []
        self.start_time = datetime.now()
        
        # 配置错误日志
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_path = os.path.join(script_dir, "access_errors.log")
        logging.basicConfig(
            filename=self.log_path,
            level=logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def _format_path(self, path):
        """转换Windows长路径格式[1,3](@ref)"""
        if platform.system() == 'Windows':
            path = os.path.abspath(path)
            if not path.startswith('\\\\?\\'):
                return '\\\\?\\' + path.lstrip('\\')
        return path

    def _update_display(self):
        """实时更新扫描进度"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        sys.stdout.write(
            f"\r📁 {self.dir_count} 文件夹 | 📝 {self.file_count} 文件 | "
            f"🗜 {self.total_size/1024/1024:.2f} MB | ⏱ {elapsed:.1f}s"
        )
        sys.stdout.flush()

    def scan(self):
        """递归扫描目录"""
        try:
            for root, dirs, files in os.walk(self.target_dir, topdown=True, onerror=self._handle_dir_error):
                self.dir_count += 1
                for file in files:
                    self._process_file(root, file)
                self._update_display()
        except KeyboardInterrupt:
            print("\n\n⚠️ 用户中断扫描")

    def _process_file(self, root, file):
        """处理单个文件（含错误捕获）[2,4](@ref)"""
        try:
            file_path = os.path.join(root, file)
            self.file_count += 1
            self.total_size += os.path.getsize(file_path)
            
            # 记录超长路径
            if len(file_path) > 255:
                self.long_paths.append(file_path)
                
        except Exception as e:
            error_msg = f"无法访问文件: {file_path} | 错误类型: {type(e).__name__}"
            logging.warning(error_msg, exc_info=True)

    def _handle_dir_error(self, error):
        """处理目录访问错误[5,7](@ref)"""
        error_path = error.filename if hasattr(error, 'filename') else '未知路径'
        error_msg = f"无法访问目录: {error_path} | 错误类型: {type(error.exc_info[1]).__name__}"
        logging.warning(error_msg, exc_info=True)

def main():
    target_dir = input("请输入要扫描的目录路径：")
    scanner = DirectoryScanner(target_dir)
    
    print("🕵️ 开始扫描...")
    scanner.scan()
    
    # 输出结果
    print("\n\n✅ 扫描完成!")
    print(f"📊 总计：{scanner.file_count} 文件 | {scanner.dir_count} 文件夹 | "
          f"{scanner.total_size/1024/1024:.2f} MB")
    
    # 输出超长路径文件
    long_paths_file = os.path.join(os.path.dirname(__file__), 'long_paths.txt')
    with open(long_paths_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(scanner.long_paths))
    
    print(f"⚠️ 找到 {len(scanner.long_paths)} 个超长路径文件，已保存至：{long_paths_file}")
    print(f"⛔ 访问错误已记录至：{scanner.log_path}")

if __name__ == '__main__':
    main()