from torch.utils.tensorboard import SummaryWriter
from uuid import uuid4

import utils.env as env_utils

class PPOLogger:
    def __init__(self, run_name=None, use_tensorboard=False):
        self.use_tensorboard = use_tensorboard
        self.global_steps = []
        self.run_name = run_name
        if self.use_tensorboard:
            run_name = str(uuid4()).hex if run_name is None else run_name
            self.writer = SummaryWriter(f"runs/{run_name}")

    def log_rollout_step(self, infos, global_step, mode):
        self.global_steps.append(global_step)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(
                        f"mode: {env_utils.mode_str_from_enum(mode)}, global_step={global_step}, episodic_return={info['episode']['r']}",
                        flush=True,
                    )

                    if self.use_tensorboard:
                        self.writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        self.writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )
            else:
                print("global_step={}".format(global_step), flush=True)
                
    def log_policy_update(self, update_results, global_step):
        if self.use_tensorboard:
            self.writer.add_scalar(
                "losses/policy_loss", update_results["policy_loss"], global_step
            )
            self.writer.add_scalar(
                "losses/value_loss", update_results["value_loss"], global_step
            )
            self.writer.add_scalar(
                "losses/entropy_loss", update_results["entropy_loss"], global_step
            )

            self.writer.add_scalar(
                "losses/kl_divergence", update_results["old_approx_kl"], global_step
            )
            self.writer.add_scalar(
                "losses/kl_divergence", update_results["approx_kl"], global_step
            )
            self.writer.add_scalar(
                "losses/clipping_fraction",
                update_results["clipping_fractions"],
                global_step,
            )
            self.writer.add_scalar(
                "losses/explained_variance",
                update_results["explained_variance"],
                global_step,
            )