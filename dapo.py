from rl_training import main


if __name__ == "__main__":
    main(
        default_loss_type="dapo",
        default_output_dir="dapo_outputs",
        default_epochs=10,
        default_beta=0.0,
    )
