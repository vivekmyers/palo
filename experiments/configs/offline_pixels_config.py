import ml_collections


def get_config(config_string):
    possible_structures = {
        "gc_iql": ml_collections.ConfigDict(
            dict(
                agent="gc_iql",
                batch_size=256,
                num_steps=int(2e6),
                log_interval=100,
                eval_interval=5000,
                save_interval=5000,
                save_dir=None,
                data_path=None,
                resume_path=None,
                seed=42,
                agent_kwargs=dict(
                    network_kwargs=dict(
                        hidden_dims=(256, 256, 256),
                    ),
                    policy_kwargs=dict(
                        tanh_squash_distribution=False,
                        state_dependent_std=False,
                        fixed_std=[1, 1, 1, 1, 1, 1, 1],
                        dropout=0.1,
                    ),
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    actor_decay_steps=int(2e6),
                    discount=0.98,
                    expectile=0.7,
                    temperature=1.0,
                    target_update_rate=0.002,
                    shared_encoder=True,
                    early_goal_concat=True,
                    shared_goal_encoder=True,
                    use_proprio=False,
                ),
                dataset_kwargs=dict(
                    shuffle_buffer_size=25000,
                    prefetch_num_batches=20,
                    relabel_actions=True,
                    goal_relabel_reached_proportion=0.1,
                    augment=True,
                    augment_next_obs_goal_differently=False,
                    augment_kwargs=dict(
                        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                        random_brightness=[0.2],
                        random_contrast=[0.8, 1.2],
                        random_saturation=[0.8, 1.2],
                        random_hue=[0.1],
                        augment_order=[
                            "random_resized_crop",
                            "random_brightness",
                            "random_contrast",
                            "random_saturation",
                            "random_hue",
                        ],
                    ),
                ),
                encoder="resnetv1-34-bridge",
                encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
            )
        ),
        "gc_bc": ml_collections.ConfigDict(
            dict(
                agent="gc_bc",
                batch_size=256,
                num_steps=int(2e6),
                log_interval=100,
                eval_interval=5000,
                save_interval=5000,
                save_dir=None,
                data_path=None,
                resume_path=None,
                seed=42,
                agent_kwargs=dict(
                    network_kwargs=dict(
                        hidden_dims=(256, 256, 256),
                    ),
                    policy_kwargs=dict(
                        tanh_squash_distribution=False,
                        fixed_std=[1, 1, 1, 1, 1, 1, 1],
                        state_dependent_std=False,
                        dropout=0.1,
                    ),
                    early_goal_concat=True,
                    shared_goal_encoder=True,
                    use_proprio=False,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    decay_steps=int(2e6),
                ),
                dataset_kwargs=dict(
                    shuffle_buffer_size=25000,
                    prefetch_num_batches=20,
                    relabel_actions=True,
                    goal_relabel_reached_proportion=0.0,
                    augment=True,
                    augment_next_obs_goal_differently=False,
                    augment_kwargs=dict(
                        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                        random_brightness=[0.2],
                        random_contrast=[0.8, 1.2],
                        random_saturation=[0.8, 1.2],
                        random_hue=[0.1],
                        augment_order=[
                            "random_resized_crop",
                            "random_brightness",
                            "random_contrast",
                            "random_saturation",
                            "random_hue",
                        ],
                    ),
                ),
                encoder="resnetv1-34-bridge",
                encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
            )
        ),
        "sg_sl": ml_collections.ConfigDict(
            dict(
                agent="multimodal",
                batch_size=64,
                num_steps=int(2e6),
                log_interval=100,
                eval_interval=5000,
                save_interval=5000,
                save_dir=None,
                data_path=None,
                resume_path=None,
                seed=42,
                agent_kwargs=dict(
                    network_kwargs=dict(
                        hidden_dims=(256, 256, 256),
                    ),
                    policy_kwargs=dict(
                        tanh_squash_distribution=False,
                        fixed_std=[1, 1, 1, 1, 1, 1, 1],
                        state_dependent_std=False,
                        dropout=0.1,
                    ),
                    early_fusion=True,
                    use_proprio=False,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    decay_steps=int(2e6),
                ),
                dataset_kwargs=dict(
                    shuffle_buffer_size=25000,
                    prefetch_num_batches=20,
                    relabel_actions=True,
                    goal_relabel_reached_proportion=0.0,
                    augment=True,
                    augment_next_obs_goal_differently=False,
                    augment_kwargs=dict(
                        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                        random_brightness=[0.2],
                        random_contrast=[0.8, 1.2],
                        random_saturation=[0.8, 1.2],
                        random_hue=[0.1],
                        augment_order=[
                            "random_resized_crop",
                            "random_brightness",
                            "random_contrast",
                            "random_saturation",
                            "random_hue",
                        ],
                    ),
                ),
                encoder="resnetv1-34-bridge",
                task_encoder="resnetv1-18-bridge",
                encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
            )
        ),
        "lc_bc": ml_collections.ConfigDict(
            dict(
                agent="multimodal",
                batch_size=256,
                num_steps=int(2e6),
                log_interval=100,
                eval_interval=5000,
                save_interval=5000,
                save_dir=None,
                data_path=None,
                resume_path=None,
                seed=42,
                agent_kwargs=dict(
                    network_kwargs=dict(
                        hidden_dims=(256, 256, 256),
                    ),
                    policy_kwargs=dict(
                        tanh_squash_distribution=False,
                        fixed_std=[1, 1, 1, 1, 1, 1, 1],
                        state_dependent_std=False,
                        dropout=0.1,
                    ),
                    early_goal_concat=True,
                    shared_goal_encoder=True,
                    use_proprio=False,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    decay_steps=int(2e6),
                ),
                dataset_kwargs=dict(
                    shuffle_buffer_size=25000,
                    prefetch_num_batches=20,
                    relabel_actions=True,
                    goal_relabel_reached_proportion=0.0,
                    augment=True,
                    augment_next_obs_goal_differently=False,
                    augment_kwargs=dict(
                        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                        random_brightness=[0.2],
                        random_contrast=[0.8, 1.2],
                        random_saturation=[0.8, 1.2],
                        random_hue=[0.1],
                        augment_order=[
                            "random_resized_crop",
                            "random_brightness",
                            "random_contrast",
                            "random_saturation",
                            "random_hue",
                        ],
                    ),
                ),
                encoder="resnetv1-34-bridge",
                encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
            )
        ),
        "sim_gc_bc": ml_collections.ConfigDict(
            dict(
                agent="gc_bc",
                batch_size=256,
                num_steps=int(2e6),
                log_interval=100,
                eval_interval=5000,
                save_interval=int(2e6),
                save_dir=None,
                data_path=None,
                resume_path=None,
                seed=42,
                env_name="PickAndPlacePositionTask",
                save_video=True,
                max_episode_steps=55,
                deterministic_eval=True,
                num_episodes_per_video=8,
                num_episodes_per_row=4,
                eval_episodes=20,
                num_val_batches=8,
                agent_kwargs=dict(
                    network_kwargs=dict(
                        hidden_dims=(256, 256, 256),
                    ),
                    policy_kwargs=dict(
                        tanh_squash_distribution=False,
                        fixed_std=[1, 1, 1, 1, 1, 1, 1],
                        state_dependent_std=False,
                        dropout=0.1,
                    ),
                    early_goal_concat=True,
                    shared_goal_encoder=True,
                    use_proprio=False,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    decay_steps=int(2e6),
                ),
                dataset_kwargs=dict(
                    shuffle_buffer_size=25000,
                    prefetch_num_batches=20,
                    goal_relabel_reached_proportion=0.0,
                    augment=True,
                    augment_next_obs_goal_differently=False,
                    augment_kwargs=dict(
                        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                        random_brightness=[0.2],
                        random_contrast=[0.8, 1.2],
                        random_saturation=[0.8, 1.2],
                        random_hue=[0.1],
                        augment_order=[
                            "random_resized_crop",
                            "random_brightness",
                            "random_contrast",
                            "random_saturation",
                            "random_hue",
                        ],
                    ),
                ),
                encoder="resnetv1-18-bridge",
                encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
            )
        ),
        "sim_gc_iql": ml_collections.ConfigDict(
            dict(
                agent="gc_iql",
                batch_size=256,
                num_steps=int(2e6),
                log_interval=100,
                eval_interval=1,
                save_interval=int(2e6),
                save_dir=None,
                data_path=None,
                resume_path=None,
                seed=42,
                env_name="PickAndPlacePositionTask",
                save_video=True,
                max_episode_steps=55,
                deterministic_eval=True,
                num_episodes_per_video=8,
                num_episodes_per_row=4,
                eval_episodes=20,
                num_val_batches=8,
                agent_kwargs=dict(
                    network_kwargs=dict(
                        hidden_dims=(256, 256, 256),
                    ),
                    policy_kwargs=dict(
                        tanh_squash_distribution=False,
                        state_dependent_std=False,
                        fixed_std=[1, 1, 1, 1, 1, 1, 1],
                        dropout=0.1,
                    ),
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    actor_decay_steps=int(2e6),
                    discount=0.98,
                    expectile=0.7,
                    temperature=1.0,
                    target_update_rate=0.002,
                    shared_encoder=True,
                    early_goal_concat=True,
                    shared_goal_encoder=True,
                    use_proprio=False,
                ),
                dataset_kwargs=dict(
                    shuffle_buffer_size=25000,
                    prefetch_num_batches=20,
                    goal_relabel_reached_proportion=0.1,
                    augment=True,
                    augment_next_obs_goal_differently=False,
                    augment_kwargs=dict(
                        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                        random_brightness=[0.2],
                        random_contrast=[0.8, 1.2],
                        random_saturation=[0.8, 1.2],
                        random_hue=[0.1],
                        augment_order=[
                            "random_resized_crop",
                            "random_brightness",
                            "random_contrast",
                            "random_saturation",
                            "random_hue",
                        ],
                    ),
                ),
                encoder="resnetv1-18-bridge",
                encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
            )
        ),
    }

    return possible_structures[config_string]
