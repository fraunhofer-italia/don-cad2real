# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

import os
import glob
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

import dense_objects_nets.utils.io_utils as io_utils
from dense_objects_nets.network.network import Network
from dense_objects_nets.dataset.don_dataset import DonDataset
from dense_objects_nets.training.don_trainer import DonTrainer
from dense_objects_nets.evaluation.don_evaluator import DonEvaluator


class DonTrainingRun():

    def __init__(self, trainer: DonTrainer, evaluator: DonEvaluator, train_dataset: DonDataset, validation_dataset: DonDataset, evaluation_dataset: DonDataset, model: Network, optimizer: Optimizer, training_config: Dict[str, Any]):
        """
        Create an object used to coordinate the Dense Object Nets training procedure.

        Args:
        - trainer (`DonTrainer`): training object used to run the training step.
        - evaluator (`DonEvaluator`): evaluator object used to run the evaluation step.
        - train_dataset (`DataLoader`): dataloader for the input training dataset (`DonDataset`) object.
        - validation_dataset (`DataLoader`): dataloader for the input validation dataset (`DonDataset`) object.
        - evaluation_dataset (`DataLoader`): dataloader for the input evaluation dataset (`DonDataset`) object.
        - model (`Network`): network model to be trained.
        - optimizer (`Optimizer`): optimizer used to update the model weights during training.
        - training_config (`dict`): dict containing the required parameters for the current training run. See 'config/training/default_training.yml` for details.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.trainer = trainer
        self.evaluator = evaluator

        self.train_data = train_dataset
        self.validation_data = validation_dataset
        self.evaluation_data = evaluation_dataset

        self.model = model
        self.optimizer = optimizer

        self.training_config = training_config

        # Set the the training run parameters.
        base_params = training_config.get("base", {})

        self.num_epochs = base_params.get("num_epochs", -1)
        self.early_stopping_patience = base_params.get("early_stopping_patience", None)
        self.epochs_between_learning_rate_decay = base_params.get("epochs_between_learning_rate_decay", self.num_epochs + 1)
        self.learning_rate_decay = base_params.get("learning_rate_decay", 1.0)

        self.validation_rate = base_params.get("validation_rate", 1.0)
        self.evaluation_rate = base_params.get("evaluation_rate", 1.0)
        self.num_valid_samples = base_params.get("num_validation_samples", 16)
        self.num_eval_samples = base_params.get("num_evaluation_samples", 16)
        self.keep_val_samples = base_params.get("keep_validation_samples", True)
        self.keep_eval_samples = base_params.get("keep_evaluation_samples", True)

        self.save_model = base_params.get("save_model", True)
        self.backup_rate = base_params.get("backup_rate", 1)
        self.keep_all = base_params.get("keep_all_backups", False)

        self.log_outputs = base_params.get("log_outputs", True)
        self.logger_queue_size = base_params.get("logger_max_queue_size", 20)
        self.images_log_rate = base_params.get("images_log_rate", 1)
        self.output_base_path = base_params.get("outputs_path", "${WS_DIR}/runs")

        # Seed the dataset if a seed is provided in the config
        seed = base_params.get("dataset_seed", None)
        if seed is not None:
            self.train_data.seed = seed
            self.validation_data.seed = seed
            self.evaluation_data.seed = seed

        # Setup the dataloaders used to wrap the training dataset (init using default values if params are not specified)
        args_names = ["batch_size", "num_workers", "pin_memory", "prefetch_factor"]
        args = {
            arg_name: base_params[arg_name]
            for arg_name in args_names if arg_name in base_params
        }
        self.train_dataloader = self.train_data.get_dataloader(**args)

    def execute(self, starting_epoch: int = 1):
        """
        Run the training procedure.

        Args:
        - starting_epoch (int, optional): index of the starting epoch for the loss functions. If training on
        a pretrained network this should be initiated to correct before training was stopped. Defaults to 0.
        and the Tensorboard log outputs will be saved. If not provided a new directory named after the current timestamp will be used.
        """
        outputs_dir = io_utils.create_timestamped_directory(
            base_path=self.output_base_path
        )

        # Save the training, dataset and evaluation config files in the output directory
        eval_config, dataset_config = self.save_configs(outputs_dir)

        patience_counter = 0
        best_valid_loss = float("inf")
        logger = SummaryWriter(log_dir=outputs_dir, max_queue=self.logger_queue_size)

        # If 'True' validation/evaluation samples are loaded only the first time and re-used at each validation/evaluation step.
        if self.keep_val_samples:
            self.validation_samples = self.validation_data.load_validation_samples(self.num_valid_samples)
        
        if self.keep_eval_samples:
            self.evaluation_samples = self.evaluation_data.load_evaluation_samples(self.num_eval_samples)

        # Start training loop
        starting_epoch = max(1, starting_epoch)  # Ensure epoch count starts at 1
        num_epochs = int(self.num_epochs) if self.num_epochs > 0 else 10**9  # Large number for infinite training
        loss_iter = (starting_epoch - 1) * len(self.train_dataloader) * self.train_dataloader.batch_size
        
        for epoch in range(starting_epoch, num_epochs + 1):
            for n_batch, data in enumerate(self.train_dataloader):
                self.model.train()
                loss_iter += self.train_dataloader.batch_size

                losses_dict = self.trainer.run(self.model, data)
                loss = self.aggregate_and_log_losses(losses_dict, logger, loss_iter)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print(f"Epoch: {epoch}, Iteration: {loss_iter}, Loss: {loss.item()}", end="\r")

                with torch.no_grad():
                    # Skip if not the last iteration of the epoch
                    # Avoid performing backup/validation/evaluation steps multiple times per epoch
                    if n_batch < len(self.train_dataloader) - 1:
                        continue

                    # Adjust the learning rate
                    self.adjust_learning_rate(epoch, logger)

                    # Periodically save the model
                    if epoch % self.backup_rate == 0:
                        self.save_network("epoch", epoch, outputs_dir, eval_config)

                    # Validation step
                    if epoch % self.validation_rate == 0:
                        self.model.eval()

                        if not self.keep_val_samples:
                            self.validation_samples = self.validation_data.load_validation_samples(
                                self.num_valid_samples
                            )

                        avg_v_losses_dict = self.trainer.run(self.model, self.validation_samples)
                        avg_v_loss = self.aggregate_and_log_losses(avg_v_losses_dict, logger, epoch, type="Validation")
                        
                        if avg_v_loss < best_valid_loss:
                            self.save_network("best", epoch, outputs_dir, eval_config)
                            best_valid_loss = avg_v_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if self.early_stopping_patience is not None and patience_counter >= self.early_stopping_patience:
                                print(f"\nEarly stopping triggered after {self.early_stopping_patience} validation epochs without improvement.")
                                break

                    # Evaluation step
                    if epoch % self.evaluation_rate == 0:
                        self.model.eval()

                        if not self.keep_eval_samples:
                            self.evaluation_samples = self.evaluation_data.load_evaluation_samples(
                                self.num_eval_samples
                            )

                        evaluation_data = self.evaluator.evaluate(
                            self.model, self.evaluation_samples
                        )

                        # Log the evaluation results on tensorboard and on file.
                        log_images = (epoch % self.images_log_rate == 0) # Whether to include images in the tensorboard log
                        self.evaluator.log_on_tensorboard(evaluation_data, logger, epoch, log_images=log_images)
                        self.evaluator.log_on_file(evaluation_data, epoch, outputs_dir)

            else:
                # Continue if the inner loop wasn't broken.
                continue
            # Inner loop was broken (early stopping), break the outer.
            break

        # Final evaluation step at the end of training
        self.evaluate_final_models(eval_config)

        # Save trainer metrics, if any
        self.trainer.log_training_metrics(outputs_dir)

        print("")

    def adjust_learning_rate(self, epoch, logger: SummaryWriter):
        """
        Adjusts the learning rate according to the schedule and logs the new value on tensorboard.

        Args:
        - epoch (`int`): current epoch number.
        - logger (`SummaryWriter`): tensorboard logger object.
        """
        if epoch % self.epochs_between_learning_rate_decay == 0:
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = param_group["lr"] * self.learning_rate_decay

        # Log the learning rate values at each epoch
        for i, param_group in enumerate(self.optimizer.param_groups):
            group_name = param_group.get("name", f"Group {i+1}")
            logger.add_scalar(f"[Learning Rates]/{group_name}", param_group["lr"], epoch)

    def aggregate_and_log_losses(self, losses: Dict[str, torch.Tensor], logger: SummaryWriter, loss_iter: int, type: str = "Train") -> torch.Tensor:
        """
        Aggregate the losses and log them on tensorboard.

        Args:
        - losses (`dict`): dict containing the loss values to be logged.
        - logger (`SummaryWriter`): tensorboard logger object.
        """
        loss = torch.tensor(0.0, device=self.device)
        for key, value in losses.items():
            # Log the loss components
            logger.add_scalar(f"{type}/{key}", value.item(), loss_iter)

            loss += value

        # if multiple components log the total loss
        if len(losses) > 1:
            logger.add_scalar(f"{type}/Loss", loss.item(), loss_iter)

        return loss

    def evaluate_final_models(self, eval_config: Dict[str, Any], max_num_samples: int = 500):
        """
        Evaluate the final models on the evaluation dataset.
        """
        if eval_config["model_best"] is None or eval_config["model_last"] is None:
            print("No models to evaluate. Skipping final evaluation.")
            return
        
        run_path = os.path.expandvars(eval_config["run_path"])
        best_model_path = os.path.join(run_path, eval_config["model_best"])
        last_model_path = os.path.join(run_path, eval_config["model_last"])
        
        model_class = self.model.__class__
        best_model = model_class.from_checkpoint(best_model_path, self.device)
        last_model = model_class.from_checkpoint(last_model_path, self.device)
        best_model.eval()
        last_model.eval()

        # Load the evaluation samples
        num_samples = min(max_num_samples, len(self.evaluation_data))
        evaluation_samples = self.evaluation_data.load_evaluation_samples(num_samples)

        best_model_eval_results = self.evaluator.evaluate(best_model, evaluation_samples)
        last_model_eval_results = self.evaluator.evaluate(last_model, evaluation_samples)

        # Log the evaluation results on file.
        self.evaluator.log_on_file(
            evaluation_results=best_model_eval_results,
            epoch="best",
            outputs_dir=run_path,
        )
        self.evaluator.log_on_file(
            evaluation_results=last_model_eval_results,
            epoch="last",
            outputs_dir=run_path,
        )

    def save_network(self, suffix: str, iteration: int, outputs_dir: str, eval_config: Dict[str, Any]):
        """
        Save the current model and optimizer configuration to file.

        Model and optimizer are saved respectively as: 'model_{suffix}_{iteration}.pth' and 'optimizer_{suffix}_{iteration}.pth'.

        If `self.keep_all` is set to `False` only the last model and optimizer with the current suffix are kept and all
        previous backups with the same suffix are overwritten. If `self.keep_all` is set to `True` all backups are saved.

        Args:
        - suffix (`str`): suffix for the current backup.
        - iteration (`int`): iteration number of the current backup.
        - outputs_dir (`str`): path to the dir where the model and optimizer state dict will be saved.
        """
        if self.keep_all == False:
            # Remove the previous model and optimizer backups with the same suffix. (If any)
            previous_backups = glob.glob(
                os.path.join(outputs_dir, f"*[model optimizer]_{suffix}_*.pth")
            )

            for backup in previous_backups:
                Path(backup).unlink(missing_ok=True)

        # Save the current model parameters
        self.model.save_checkpoint(
            filepath=os.path.join(outputs_dir, f"model_{suffix}_{iteration}.pth")
        )

        # Save the current optimizer parameters
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(outputs_dir, f"optimizer_{suffix}_{iteration}.pth"),
        )

        # Update the evaluation config
        key = "model_best" if suffix == "best" else "model_last"
        eval_config[key] = f"model_{suffix}_{iteration}.pth"
        io_utils.save_dict(eval_config, name="evaluation_config.yml", path=outputs_dir)


    def save_configs(self, outputs_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Save the training, dataset and evaluation config files in the output directory.
        Args:
        - outputs_dir (`str`): path to the output directory where config files will be saved.
        """
        dataset_config = {
            "train": self.train_data.scenes_config,
            "validation": self.validation_data.scenes_config,
            "evaluation": self.evaluation_data.scenes_config,
        }

        eval_config = {
            "evaluator_class": f"{self.evaluator.__class__.__module__}.{self.evaluator.__class__.__name__}",
            "dataset_class": f"{self.evaluation_data.__class__.__module__}.{self.evaluation_data.__class__.__name__}",
            "trainer_class": f"{self.trainer.__class__.__module__}.{self.trainer.__class__.__name__}",
            "network_class": f"{self.model.__class__.__module__}.{self.model.__class__.__name__}",
            "transforms": [t.__class__.__name__ for t in self.train_data.transforms],
            "run_path": outputs_dir.replace(os.path.expandvars("${WS_DIR}"), "${WS_DIR}"),
            "model_best": None,
            "model_last": None,
            "plot_object_only": False,
            "evaluation_split": "evaluation", # Can be either 'train', 'validation' or 'evaluation'
        }

        io_utils.save_dict(self.training_config, name="training_config.yml", path=outputs_dir)
        io_utils.save_dict(dataset_config, name="dataset_config.yml", path=outputs_dir)
        io_utils.save_dict(eval_config, name="evaluation_config.yml", path=outputs_dir)

        return eval_config, dataset_config
