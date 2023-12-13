def train():
    import torch
    import gc
    import os
    import subprocess
    import plot_grid
    import accelerate
    torch.cuda.empty_cache()
    gc.collect()  # Force garbage collection

    for i in range(2, 22, 2):
        with torch.no_grad():
            subprocess.run(['accelerate', 'config', 'default', '--mixed_precision', 'fp16'])
            subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv'])
    
            os.environ['MODEL_NAME'] = 'CompVis/stable-diffusion-v1-4'
            os.environ['INSTANCE_DIR'] = './bash_images'
            os.environ['OUTPUT_DIR'] = f'./dreambooth/run{i}'
            os.environ['TRAIN_STEPS'] = str(i * 50)
    
            print(i * 50)
            subprocess.run([
                'accelerate', 'launch', './diffusers/examples/dreambooth/train_dreambooth.py',
                '--pretrained_model_name_or_path', os.environ['MODEL_NAME'],
                '--instance_data_dir', os.environ['INSTANCE_DIR'],
                '--output_dir', os.environ['OUTPUT_DIR'],
                '--instance_prompt', 'a photo of brk1',
                '--resolution', '512',
                '--train_batch_size', '1',
                '--gradient_accumulation_steps', '1',
                '--learning_rate', '5e-6',
                '--lr_scheduler', 'constant',
                '--lr_warmup_steps', '0',
                '--max_train_steps', os.environ['TRAIN_STEPS']
            ])
    
            # model_id = f"./dreambooth/run{i}"
            # grid = plot_grid.plotting_grid(model_id)
            # grid.save(f'./grid_run_{i}.png')
        os.environ.clear()
        # del grid, model_id
        locals().clear()

if __name__ == "__main__":
    train()
