import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import threading
import torch
from gui.main_window import MainWindow
from models.model_manager import ModelManager
from converters.dicom2nii import DicomConverter, DatasetSplitter
import PySimpleGUI as sg
from models.UNet3Plus import UNet3Plus


class Application:
    def __init__(self):
        self.window = MainWindow(
            canvas_key='-TRAIN_CANVAS-',
            progress_key='-PROGRESS-',
            status_keys=('-CURRENT_EPOCH-', '-BEST_LOSS-')
        )
        self.model_mgr = ModelManager()
        self.model_mgr.register_model('UNet3Plus', UNet3Plus)  # 注册模型
        self.converter = DicomConverter()
        self.is_training = False
    
    def run(self):
        while True:
            event, values = self.window.window.read()  
            
            if event == sg.WIN_CLOSED:
                break
                
            if event == '-CONVERT_DICOM-':
                threading.Thread(target=self.convert_dicom, args=(values['-DICOM_DIR-'], 
                                values['-NII_DIR-']), daemon=True).start()
                
            if event == '-APPLY_PARAMS-':
                self.apply_model_params(values)
                
            if event == '-START_TRAIN-':
                self.start_training()
            
            if event == '-STATUS_UPDATE-':
                self.window.window['-STATUS-'].update(values[event])
        
        self.window.window.close()
    
    def convert_dicom(self, input_dir, output_dir):
        try:
            self.converter.convert_series(input_dir, output_dir)
            self.window.window.write_event_value('-STATUS_UPDATE-', '转换完成')
        except Exception as e:
            self.window.window['-STATUS-'].update(f'转换错误: {str(e)}')
    
    def apply_model_params(self, values):
        config = {
            'in_channels': int(values['-IN_CH-']),
            'out_channels': int(values['-NUM_CLASS-']),
            'feature_scale': int(values['-FEATURE_SCALE-']),
            'is_batchnorm': values['-USE_BN-'],
            'train_data': values['-TRAIN_DATA-'],
            'labels': values['-LABELS-'],
            'epochs': int(values['-EPOCHS-']),         # 新增
            'batch_size': int(values['-BATCH_SIZE-'])  # 新增
        }
        self.train_epochs = config['epochs']
        self.batch_size = config['batch_size']
        model_name = 'UNet3Plus'
        self.model_mgr.initialize_model(model_name, **config)
        self.window.window['-STATUS-'].update('模型初始化完成')
    
    def start_training(self):
        if not self.model_mgr.current_model:
            self.window.window['-STATUS-'].update('请先初始化模型')
            return
        if self.is_training:
            self.window.window['-STATUS-'].update('训练已在进行中')
            return
        self.is_training = True
        self.window.window['-STATUS-'].update('训练启动...')
        # 启动训练线程
        threading.Thread(target=self.train_loop, daemon=True).start()

    def train_loop(self):
        num_epochs = self.train_epochs  # 使用apply_model_params中保存的值
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate_one_epoch(epoch)
            self.window.monitor.update_metrics(
                train_loss=train_loss,
                val_loss=val_loss,
                current_epoch=epoch,
                total_epochs=num_epochs,
                best_loss=min(self.window.monitor.val_loss) if self.window.monitor.val_loss else None
            )
            self.window.window.write_event_value('-STATUS_UPDATE-', f'训练进度: {epoch}/{num_epochs}')
        self.is_training = False
        self.window.window.write_event_value('-STATUS_UPDATE-', '训练完成')

    def train_one_epoch(self, epoch):
        # 这里写单轮训练逻辑，返回loss
        return 1.0 / epoch  # 示例

    def validate_one_epoch(self, epoch):
        # 这里写单轮验证逻辑，返回loss
        return 1.2 / epoch  # 示例

if __name__ == '__main__':
    app = Application()
    app.run()