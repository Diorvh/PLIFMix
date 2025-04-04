import json, os
import tkinter as tk

class ConfigManager:
    def __init__(self, root, config_name):
        self.root = root
        self.config_name = config_name+'-config.json'
        self.save_entry = {}
        self.save_var = {}

    def register_var(self, name, var):
        self.save_var[name] = var
    
    def register_entry(self, name, entry):
        self.save_entry[name] = entry

    def save_config(self):
        """保存配置"""
        config = {
            'var': {name: var.get() for name, var in self.save_var.items()},
            'entry': {name: entry.get() for name, entry in self.save_entry.items()}
        }
        with open(self.config_name,'w') as f:
            json.dump(config, f)
    
    def load_config(self):
        """加载配置"""
        if os.path.exists(self.config_name):
            with open(self.config_name,'r') as f:
                config = json.load(f)
                for name, value in config['var'].items():
                    if isinstance(self.save_var[name],tk.BooleanVar):
                        self.save_var[name].set(True if value == 'True' else False)
                    else:
                        self.save_var[name].set(value)
                for name, value in config['entry'].items():
                    self.save_entry[name].insert(0, value)