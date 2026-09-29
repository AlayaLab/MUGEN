import os, time, logging, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def _setup_training_logger(output_path):
    """配置训练日志，同时输出到文件和 stdout。"""
    logger = logging.getLogger("wan360_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    # stdout
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    # file
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_path, "train.log"), mode="a")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs

    logger = _setup_training_logger(model_logger.output_path if accelerator.is_main_process else None)

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, collate_fn=lambda x: x[0],
        num_workers=num_workers, pin_memory=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None,
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    total_steps = len(dataloader) * num_epochs
    if accelerator.is_main_process:
        logger.info(f"Training started | epochs={num_epochs} steps_per_epoch={len(dataloader)} total_steps={total_steps} lr={learning_rate} num_workers={num_workers}")

    global_step = 0
    for epoch_id in range(num_epochs):
        epoch_loss_sum = 0.0
        epoch_steps = 0
        step_start = time.time()

        for data in tqdm(dataloader, desc=f"Epoch {epoch_id}", disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()

            global_step += 1
            epoch_steps += 1
            loss_val = loss.item()
            epoch_loss_sum += loss_val
            step_time = time.time() - step_start

            if accelerator.is_main_process and global_step % 10 == 1:
                gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
                logger.info(
                    f"step={global_step}/{total_steps} epoch={epoch_id} "
                    f"loss={loss_val:.6f} avg_loss={epoch_loss_sum/epoch_steps:.6f} "
                    f"step_time={step_time:.2f}s gpu_mem={gpu_mem:.1f}GB"
                )

            step_start = time.time()

        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch_id} done | avg_loss={epoch_loss_sum/max(epoch_steps,1):.6f} steps={epoch_steps}")

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)

    if accelerator.is_main_process:
        logger.info("Training finished.")


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
