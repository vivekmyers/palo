import ml_collections
import flax.linen as nn

PNP_TASKS = [
    "bridge_data_v1/berkeley/toysink1_room8052/put_pan_from_sink_into_drying_rack/",
    "bridge_data_v1/berkeley/toysink1_room8052/put_pan_from_drying_rack_into_sink/",
    "bridge_data_v1/berkeley/toysink1_room8052/put_pan_on_stove_from_sink/",
    "bridge_data_v1/berkeley/toysink1_room8052/put_spoon_into_pan/",
    "bridge_data_v1/berkeley/toysink1_room8052/put_pan_from_stove_to_sink/",
    "bridge_data_v1/berkeley/toysink1_room8052/put_eggplant_into_pan/",
    "bridge_data_v1/berkeley/realkitchen1_counter/put_spoon_on_plate/",
    "bridge_data_v1/berkeley/realkitchen1_counter/pick_up_sponge_and_wipe_plate/",
    "bridge_data_v1/berkeley/realkitchen1_dishwasher/pick_up_any_cup/",
    "bridge_data_v1/berkeley/realkitchen1_dishwasher/pick_up_green_mug/",
    "bridge_data_v1/berkeley/realkitchen1_dishwasher/pick_up_glass_cup/",
    "bridge_data_v1/berkeley/toysink2_bww/put_carrot_on_plate/",
    "bridge_data_v1/berkeley/toysink2_bww/put_spoon_in_pot/",
    "bridge_data_v1/berkeley/toysink2_bww/put_knife_on_cutting_board/",
    "bridge_data_v1/berkeley/toysink2_bww/put_cup_from_counter_or_drying_rack_into_sink/",
    "bridge_data_v1/berkeley/toysink2_bww/put_eggplant_into_pot_or_pan/",
    "bridge_data_v1/berkeley/toykitchen4/put_banana_in_pot_or_pan/",
    "bridge_data_v1/berkeley/toykitchen4/put_lid_on_pot_or_pan/",
    "bridge_data_v1/berkeley/toykitchen4/put_pear_on_plate/",
    "bridge_data_v1/berkeley/toykitchen4/put_carrot_in_bowl/",
    "bridge_data_v1/berkeley/toykitchen4/put_sushi_in_pot_or_pan/",
    "bridge_data_v1/berkeley/toykitchen4/put_detergent_in_sink/",
    "bridge_data_v1/berkeley/laundry_machine/take_clothes_out_of_laundry_machine/",
    "bridge_data_v1/berkeley/laundry_machine/put_clothes_in_laundry_machine/",
    "bridge_data_v1/berkeley/toysink3_bww/put_cup_into_pot_or_pan/",
    "bridge_data_v1/berkeley/toysink3_bww/put_lid_on_pot_or_pan/",
    "bridge_data_v1/berkeley/toysink3_bww/put_cup_from_anywhere_into_sink/",
    "bridge_data_v1/berkeley/toysink3_bww/put_knife_in_pot_or_pan/",
    "bridge_data_v1/berkeley/toysink3_bww/put_green_squash_into_pot_or_pan/",
    "bridge_data_v1/berkeley/toysink3_bww/take_lid_off_pot_or_pan/",
    "bridge_data_v1/berkeley/toysink3_bww/put_pot_or_pan_from_sink_into_drying_rack/",
    "bridge_data_v1/berkeley/toysink3_bww/put_brush_into_pot_or_pan/",
    "bridge_data_v1/berkeley/toysink3_bww/put_detergent_from_sink_into_drying_rack/",
    "bridge_data_v1/berkeley/toykitchen1/put_sushi_on_plate/",
    "bridge_data_v1/berkeley/toykitchen1/put_pan_in_sink/",
    "bridge_data_v1/berkeley/toykitchen1/put_broccoli_in_bowl/",
    "bridge_data_v1/berkeley/toykitchen1/put_pot_on_stove_which_is_near_stove_distractors/",
    "bridge_data_v1/berkeley/toykitchen1/put_corn_into_bowl/",
    "bridge_data_v1/berkeley/toykitchen1/take_can_out_of_pan/",
    "bridge_data_v1/berkeley/toykitchen1/put_fork_from_basket_to_tray/",
    "bridge_data_v1/berkeley/toykitchen1/put_eggplant_on_plate/",
    "bridge_data_v1/berkeley/toykitchen1/put_lid_on_pot_or_pan/",
    "bridge_data_v1/berkeley/toykitchen1/put_corn_in_pot_which_is_in_sink_distractors/",
    "bridge_data_v1/berkeley/toykitchen1/put_carrot_on_plate/",
    "bridge_data_v1/berkeley/toykitchen1/take_carrot_off_plate_cardboardfence/",
    "bridge_data_v1/berkeley/toykitchen1/put_sweet_potato_in_pan_which_is_on_stove_distractors/",
    "bridge_data_v1/berkeley/toykitchen1/put_big_spoon_from_basket_to_tray/",
    "bridge_data_v1/berkeley/toykitchen1/take_broccoli_out_of_pan/",
    "bridge_data_v1/berkeley/toykitchen1/put_lid_on_stove/",
    "bridge_data_v1/berkeley/toykitchen1/put_sweet_potato_in_pan_which_is_on_stove/",
    "bridge_data_v1/berkeley/toykitchen1/put_corn_in_pan_which_is_on_stove_distractors/",
    "bridge_data_v1/berkeley/toykitchen1/put_pear_in_bowl/",
    "bridge_data_v1/berkeley/toykitchen1/pick_up_pan_from_stove_distractors/",
    "bridge_data_v1/berkeley/toykitchen1/put_banana_on_plate/",
    "bridge_data_v1/berkeley/toykitchen1/pick_up_pot_from_sink_distractors/",
    "bridge_data_v1/berkeley/toykitchen1/put_sweet_potato_in_pot_which_is_in_sink_distractors/",
    "bridge_data_v1/berkeley/toykitchen1/put_pot_in_sink/",
    "bridge_data_v1/berkeley/toykitchen1/take_sushi_out_of_pan/",
    "bridge_data_v1/berkeley/toykitchen1/put_detergent_in_sink/",
    "bridge_data_v1/berkeley/toykitchen1/take_broccoli_out_of_pan_cardboardfence/",
    "bridge_data_v1/berkeley/toykitchen1/put_broccoli_in_pot_cardboardfence/",
    "bridge_data_v1/berkeley/toykitchen1/take_lid_off_pot_or_pan/",
    "bridge_data_v1/berkeley/toykitchen1/take_carrot_off_plate/",
    "bridge_data_v1/berkeley/toykitchen1/put_eggplant_in_pot_or_pan/",
    "bridge_data_v1/berkeley/toykitchen1/put_carrot_on_plate_cardboardfence/",
    "bridge_data_v1/berkeley/toykitchen1/pick_up_bowl_and_put_in_small4fbox/",
    "bridge_data_v1/berkeley/toykitchen1/put_carrot_on_cutting_board/",
    "bridge_data_v1/berkeley/toykitchen1/put_red_bottle_in_sink/",
    "bridge_data_v1/berkeley/toykitchen1/put_pepper_in_pan/",
    "bridge_data_v1/berkeley/toykitchen1/put_broccoli_in_pot_or_pan/",
    "bridge_data_v1/berkeley/toykitchen1/put_knife_on_cutting_board/",
    "bridge_data_v1/berkeley/toykitchen1/put_small_spoon_from_basket_to_tray/",
    "bridge_data_v1/berkeley/toykitchen1/put_corn_in_pan_which-is_on_stove_distractors/",
    "bridge_data_v1/berkeley/toykitchen1/put_pepper_in_pot_or_pan/",
    "bridge_data_v1/berkeley/toykitchen1/put_green_squash_in_pot_or_pan/",
    "bridge_data_v1/berkeley/tabletop_dark_wood/put_spatula_on_cutting_board/",
    "bridge_data_v1/berkeley/tabletop_dark_wood/put_banana_in_colander/",
    "bridge_data_v1/berkeley/tabletop_dark_wood/take_banana_out_of_colander/",
    "bridge_data_v1/berkeley/toykitchen6/take_cup_off_plate/",
    "bridge_data_v1/berkeley/toykitchen6/put_spatula_on_plate_sink/",
    "bridge_data_v1/berkeley/toykitchen6/take_spoon_out_of_bowl_sink/",
    "bridge_data_v1/berkeley/toykitchen6/put_beet_in_pot_sink/",
    "bridge_data_v1/berkeley/toykitchen6/put_corn_in_bowl_sink/",
    "bridge_data_v1/berkeley/toykitchen6/put_blueberries_on_plate_sink/",
    "bridge_data_v1/berkeley/toykitchen6/take_corn_out_of_bowl_sink/",
    "bridge_data_v1/berkeley/toykitchen6/put_cup_on_plate/",
    "bridge_data_v1/berkeley/toykitchen6/take_blueberries_off_plate_sink/",
    "bridge_data_v1/berkeley/toykitchen6/take_beet_from_pot_sink/",
    "bridge_data_v1/berkeley/toykitchen6/take_spatula_off_plate_sink/",
    "bridge_data_v1/berkeley/toykitchen6/put_spoon_in_bowl_sink/",
    "bridge_data_v1/berkeley/tabletop_white/put_sushi_on_plate/",
    "bridge_data_v1/berkeley/tabletop_white/take_sushi_off_plate/",
    "bridge_data_v1/berkeley/tabletop_light_wood/put_cucumber_in_cup/",
    "bridge_data_v1/berkeley/tabletop_light_wood/take_cucumber_out_of_cup/",
    "bridge_data_v1/berkeley/toykitchen2/take_bowl_off_plate_cardboard_fence/",
    "bridge_data_v1/berkeley/toykitchen2/put_bowl_on_plate/",
    "bridge_data_v1/berkeley/toykitchen2/take_bowl_off_plate/",
    "bridge_data_v1/berkeley/toykitchen2/take_sushi_out_of_pot_cardboard_fence/",
    "bridge_data_v1/berkeley/toykitchen2/take_lid_off_pot_cardboardfence/",
    "bridge_data_v1/berkeley/toykitchen2/put_potato_in_pot_cardboard_fence/",
    "bridge_data_v1/berkeley/toykitchen2/take_carrot_out_of_pot_cardboard_fence/",
    "bridge_data_v1/berkeley/toykitchen2/put_carrot_in_pot_cardboard_fence/",
    "bridge_data_v1/berkeley/toykitchen2/put_knife_on_cutting_board_cardboard_fence/",
    "bridge_data_v1/berkeley/toykitchen2/put_cap_on_container/",
    "bridge_data_v1/berkeley/toykitchen2/put_sushi_in_pot_cardboard_fence/",
    "bridge_data_v1/berkeley/toykitchen2/put_bowl_on_plate_cardboard_fence/",
    "bridge_data_v1/berkeley/toykitchen2/put_lid_on_pot_cardboardfence/",
    "bridge_data_v1/berkeley/toykitchen2/put_banana_in_pot_cardboard_fence/",
    "bridge_data_v1/berkeley/toykitchen2/put_pear_in_bowl_cardboardfence/",
    "bridge_data_v1/berkeley/toykitchen2/put_knife_in_pot_cardboard_fence/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/put_spatula_in_pan/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/put_pot_or_pan_on_stove/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/put_potato_in_pot_or_pan/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/put_pear_in_bowl/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/put_knife_on_cutting_board/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/put_pot_or_pan_in_sink/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/put_strawberry_in_pot/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/put_lemon_on_plate/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/put_corn_on_plate/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/put_sushi_on_plate/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/put_can_in_pot/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/put_potato_on_plate/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/lift_bowl/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/put_carrot_in_pot_or_pan/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/put_sweet_potato_in_pot/",
    "bridge_data_v1/berkeley/tool_chest/pick_up_blue_pen_and_put_into_drawer/",
    "bridge_data_v1/berkeley/tool_chest/pick_up_red_srewdriver/",
    "bridge_data_v1/berkeley/tool_chest/pick_up_box_cutter_and_put_into_drawer/",
    "bridge_data_v1/berkeley/tool_chest/pick_up_violet_Allen_key/",
    "bridge_data_v1/berkeley/tool_chest/pick_up_bit_holder/",
    "bridge_data_v1/berkeley/tool_chest/pick_up_scissors_and_put_into_drawer/",
    "bridge_data_v1/berkeley/tool_chest/pick_up_glue_and_put_into_drawer/",
]

multimodal_config_proto = dict(
    agent="multimodal",
    batch_size=128,
    num_steps=int(3e5),
    log_interval=100,
    eval_interval=5000,
    save_interval=5000,
    save_dir=None,
    data_path=None,
    num_annotations=None,
    resume_path=None,
    augment_language=True,
    clip_resume_path=None,
    use_image_embeddings=False,
    use_text_embeddings=False,
    use_image_embeds_as_inputs=False,
    use_text_embeds_as_inputs=False,
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
        alignment=0.0,
        flatten_task_reps=False,
    ),
    domain_weight=None,
    dataset_kwargs=dict(
        clip_preprocessing=False,
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
    ss2_train_path=None,
    ss2_val_path=None,
    ss2_labels_path=None,
    ss2_batch_size=0,
    ss2_val_batch_size=128,
    ss2_dataset_kwargs=dict(
        shuffle_buffer_size=25,
        prefetch_num_batches=20,
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
    encoder="resnetv1-34-bridge-task-x-units",
    encoder_kwargs=dict(
        pooling_method="avg",
        add_spatial_coordinates=True,
        act="swish",
        task_units=256,
    ),
    task_encoder_kwargs=dict(),
)


def update_config(_prototype=multimodal_config_proto, **kwargs):
    result = dict(_prototype)
    for key, value in kwargs.items():
        if (
            type(result.get(key)) == dict
            or type(result.get(key)) == ml_collections.ConfigDict
        ):
            if kwargs[key].get("_overwrite", False) is False:
                value = dict(update_config(_prototype=result[key], **kwargs[key]))
            value.pop("_overwrite", None)
        result[key] = value
    result.pop("_overwrite", None)
    return ml_collections.ConfigDict(result)


def get_config(config_string):
    possible_structures = {
        "all": ml_collections.ConfigDict(
            {
                "include": [
                    [
                        "icra/?*/?*/?*",
                        "icra_validation/?*/?*/?*",
                        "flap/?*/?*/?*",
                        "bridge_data_v1/berkeley/?*/?*",
                        "rss/?*/?*/?*",
                        "bridge_data_v2/?*/?*/?*",
                        "scripted/?*",
                    ]
                ],
                "exclude": [
                    "*toykitchen7*",
                    "*tabletop_dark_wood*",
                    "*icra_validation/toykitchen_fixed_cam_offline_validation/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree_combo/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree_push_sweep/tabletop*",
                    "*sweep_12-03*",
                ],
                "sample_weights": None,
                "action_metadata": {
                    "mean": [
                        0.0000799,
                        0.00029883,
                        -0.00014265,
                        -0.00007046,
                        -0.00000396,
                        0.00008324,
                        0.5428687,
                    ],
                    "std": [
                        0.0073178,
                        0.01133177,
                        0.01211691,
                        0.01779856,
                        0.02181558,
                        0.08579306,
                        0.49777785,
                    ],
                },
            }
        ),
        "no_rss_toykitchen2": ml_collections.ConfigDict(
            {
                "include": [
                    [
                        "icra/*/*/*",
                        "icra_validation/?*/?*/?*",
                        "flap/?*/?*/?*",
                        "bridge_data_v1/berkeley/?*/?*",
                        "rss/toykitchen6/?*/?*",
                    ]
                ],
                "exclude": [
                    "toykitchen7",
                    "tabletop_dark_wood",
                    "icra_validation/toykitchen_fixed_cam_offline_validation/tabletop",
                    "icra/toykitchen_fixed_cam_resetfree/tabletop",
                    "icra/toykitchen_fixed_cam_resetfree_combo/tabletop",
                    "icra/toykitchen_fixed_cam_resetfree_push_sweep/tabletop",
                ],
                "sample_weights": None,
                "action_metadata": {
                    "mean": [
                        0.0000799,
                        0.00029883,
                        -0.00014265,
                        -0.00007046,
                        -0.00000396,
                        0.00008324,
                        0.5428687,
                    ],
                    "std": [
                        0.0073178,
                        0.01133177,
                        0.01211691,
                        0.01779856,
                        0.02181558,
                        0.08579306,
                        0.49777785,
                    ],
                },
            }
        ),
        "no_scripted": ml_collections.ConfigDict(
            {
                "include": [
                    [
                        "icra/*/*/*",
                        "icra_validation/?*/?*/?*",
                        "flap/?*/?*/?*",
                        "bridge_data_v1/berkeley/?*/?*",
                        "rss/*?/?*/?*",
                    ]
                ],
                "exclude": [
                    "toykitchen7",
                    "scripted",
                    "tabletop_dark_wood",
                    "icra_validation/toykitchen_fixed_cam_offline_validation/tabletop",
                    "icra/toykitchen_fixed_cam_resetfree/tabletop",
                    "icra/toykitchen_fixed_cam_resetfree_combo/tabletop",
                    "icra/toykitchen_fixed_cam_resetfree_push_sweep/tabletop",
                ],
                "sample_weights": None,
                "action_metadata": {
                    "mean": [
                        0.0000799,
                        0.00029883,
                        -0.00014265,
                        -0.00007046,
                        -0.00000396,
                        0.00008324,
                        0.5428687,
                    ],
                    "std": [
                        0.0073178,
                        0.01133177,
                        0.01211691,
                        0.01779856,
                        0.02181558,
                        0.08579306,
                        0.49777785,
                    ],
                },
            }
        ),
        "all_finetune": ml_collections.ConfigDict(
            {
                "include": [
                    [
                        "icra/*/*/*",
                        "icra_validation/?*/?*/?*",
                        "flap/?*/?*/?*",
                        "bridge_data_v1/berkeley/?*/?*",
                        "rss/toykitchen2/?*/?*",
                        "rss/toykitchen6/?*/?*",
                    ],
                    ["rss/toykitchen7/pnp_sweep_target_fixed/?*"],
                ],
                "exclude": [
                    "*tabletop_dark_wood*",
                    "*icra_validation/toykitchen_fixed_cam_offline_validation/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree_combo/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree_push_sweep/tabletop*",
                ],
                "sample_weights": [0.9, 0.1],
                "action_metadata": {
                    "mean": [
                        0.0000799,
                        0.00029883,
                        -0.00014265,
                        -0.00007046,
                        -0.00000396,
                        0.00008324,
                        0.5428687,
                    ],
                    "std": [
                        0.0073178,
                        0.01133177,
                        0.01211691,
                        0.01779856,
                        0.02181558,
                        0.08579306,
                        0.49777785,
                    ],
                },
            }
        ),
        "all_finetune_autonomous": ml_collections.ConfigDict(
            {
                "include": [
                    [
                        "icra/?*/?*/?*",
                        "icra_validation/?*/?*/?*",
                        "flap/?*/?*/?*",
                        "bridge_data_v1/berkeley/?*/?*",
                        "rss/?*/?*/?*",
                        "bridge_data_v2/?*/?*/?*",
                        "scripted/?*",
                    ],
                    ["learned/?*/?*"],
                ],
                "exclude": [
                    "*rss/toykitchen7*",
                    "*tabletop_dark_wood*",
                    "*icra_validation/toykitchen_fixed_cam_offline_validation/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree_combo/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree_push_sweep/tabletop*",
                    "*sweep_12-03*",
                ],
                "sample_weights": [0.9, 0.1],
                "action_metadata": {
                    "mean": [
                        0.0000799,
                        0.00029883,
                        -0.00014265,
                        -0.00007046,
                        -0.00000396,
                        0.00008324,
                        0.5428687,
                    ],
                    "std": [
                        0.0073178,
                        0.01133177,
                        0.01211691,
                        0.01779856,
                        0.02181558,
                        0.08579306,
                        0.49777785,
                    ],
                },
            }
        ),
        "all_finetune_autonomous_oracle": ml_collections.ConfigDict(
            {
                "include": [
                    [
                        "icra/*/*/*",
                        "icra_validation/?*/?*/?*",
                        "flap/?*/?*/?*",
                        "bridge_data_v1/berkeley/?*/?*",
                        "rss/toykitchen2/?*/?*",
                        "rss/toykitchen6/?*/?*",
                    ],
                    [
                        "finetuning/ours_2_22/?*",
                        "rss/toykitchen7/pnp_sweep_target_fixed/?*",
                    ],
                ],
                "exclude": [
                    "*tabletop_dark_wood*",
                    "*icra_validation/toykitchen_fixed_cam_offline_validation/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree_combo/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree_push_sweep/tabletop*",
                ],
                "sample_weights": [0.9, 0.1],
                "action_metadata": {
                    "mean": [
                        0.0000799,
                        0.00029883,
                        -0.00014265,
                        -0.00007046,
                        -0.00000396,
                        0.00008324,
                        0.5428687,
                    ],
                    "std": [
                        0.0073178,
                        0.01133177,
                        0.01211691,
                        0.01779856,
                        0.02181558,
                        0.08579306,
                        0.49777785,
                    ],
                },
            }
        ),
        "settable-scripted_bridge_pnp": ml_collections.ConfigDict(
            {
                "include": [
                    [
                        "rss/toykitchen2/?*/?*",
                        "rss/toykitchen6/?*/?*",
                    ],
                    [
                        "icra/?*/?*/?*",
                        "flap/?*/?*/?*",
                        "icra_validation/?*/?*/?*",
                    ]
                    + PNP_TASKS,
                ],
                "exclude": [
                    "*toykitchen7*",
                    "*tabletop_dark_wood*",
                    "*icra_validation/toykitchen_fixed_cam_offline_validation/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree_combo/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree_push_sweep/tabletop*",
                ],
                "sample_weights": [0.7, 0.3],
                "action_metadata": {
                    "mean": [
                        0.0000799,
                        0.00029883,
                        -0.00014265,
                        -0.00007046,
                        -0.00000396,
                        0.00008324,
                        0.5428687,
                    ],
                    "std": [
                        0.0073178,
                        0.01133177,
                        0.01211691,
                        0.01779856,
                        0.02181558,
                        0.08579306,
                        0.49777785,
                    ],
                },
            }
        ),
        "settable-scripted_bridge_pnp_finetune": ml_collections.ConfigDict(
            {
                "include": [
                    [
                        "rss/toykitchen2/?*/?*",
                        "rss/toykitchen6/?*/?*",
                        "icra/?*/?*/?*",
                        "flap/?*/?*/?*",
                        "icra_validation/?*/?*/?*",
                    ]
                    + PNP_TASKS,
                    ["rss/toykitchen7/pnp_sweep_target_fixed/?*"],
                ],
                "exclude": [
                    "*tabletop_dark_wood*",
                    "*icra_validation/toykitchen_fixed_cam_offline_validation/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree_combo/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree_push_sweep/tabletop*",
                ],
                "sample_weights": [0.9, 0.1],
                "action_metadata": {
                    "mean": [
                        0.0000799,
                        0.00029883,
                        -0.00014265,
                        -0.00007046,
                        -0.00000396,
                        0.00008324,
                        0.5428687,
                    ],
                    "std": [
                        0.0073178,
                        0.01133177,
                        0.01211691,
                        0.01779856,
                        0.02181558,
                        0.08579306,
                        0.49777785,
                    ],
                },
            }
        ),
        "sg_sl_align": update_config(
            task_encoders=dict(
                image="resnetv1-18-bridge", language="resnetv1-18-bridge-task"
            ),
            task_encoder_kwargs=dict(
                image=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
                language=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
            ),
            agent_kwargs=dict(
                alignment=1.0,
            ),
        ),
        "sg_sl_align_clip_frozen_with_mlp": update_config(
            clip_resume_path=None,
            task_encoders=dict(
                image="clip_vision_with_projection",
                language="clip_text_with_projection",
            ),
            task_encoder_kwargs=dict(
                image=dict(
                    freeze_encoder=True,
                    mlp_kwargs=dict(
                        hidden_dims=(256,),
                        activation=nn.relu,
                    ),
                ),
                language=dict(
                    freeze_encoder=True,
                    mlp_kwargs=dict(
                        hidden_dims=(256,),
                        activation=nn.relu,
                    ),
                ),
            ),
            agent_kwargs=dict(
                alignment=1.0,
            ),
        ),
        "sg_sl_align_clip_frozen_tune_projection": update_config(
            clip_resume_path=None,
            task_encoders=dict(
                image="clip_vision_with_projection",
                language="clip_text_with_projection",
            ),
            task_encoder_kwargs=dict(
                image=dict(
                    freeze_encoder=True,
                ),
                language=dict(
                    freeze_encoder=True,
                ),
            ),
            agent_kwargs=dict(
                alignment=1.0,
            ),
        ),
        "sg_sl_align_clip_lang_frozen_tune_projection": update_config(
            clip_resume_path=None,
            task_encoders=dict(
                image="resnetv1-18-bridge",
                language="clip_text_with_projection",
            ),
            task_encoder_kwargs=dict(
                image=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
                language=dict(
                    freeze_encoder=True,
                ),
            ),
            agent_kwargs=dict(
                alignment=1.0,
            ),
        ),
        "sg_sl_align_clip_lang_frozen": update_config(
            clip_resume_path=None,
            task_encoders=dict(
                image="resnetv1-18-bridge",
                language="clip_text_with_projection",
            ),
            task_encoder_kwargs=dict(
                image=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                    mlp_kwargs=dict(
                        hidden_dims=(
                            128,
                            512,
                        ),
                        activation=nn.relu,
                    ),
                ),
                language=dict(
                    freeze_encoder=True,
                    freeze_projection=True,
                ),
            ),
            agent_kwargs=dict(
                alignment=1.0,
            ),
        ),
        "sg_sl_align_clip_img_tune": update_config(
            clip_resume_path=None,
            task_encoders=dict(
                image="clip_vision_with_projection",
                language="clip_text_with_projection",
            ),
            task_encoder_kwargs=dict(
                image=dict(
                    freeze_encoder=False,
                ),
                language=dict(
                    freeze_encoder=True,
                    freeze_projection=True,
                ),
            ),
            agent_kwargs=dict(
                alignment=0.0,
            ),
        ),
        "sg_sl_align_clip_frozen": update_config(
            clip_resume_path=None,
            task_encoders=dict(
                image="clip_vision_with_projection",
                language="clip_text_with_projection",
            ),
            task_encoder_kwargs=dict(
                image=dict(
                    freeze_encoder=True,
                    mlp_kwargs=dict(
                        hidden_dims=(256, 512),
                        activation=nn.relu,
                    ),
                ),
                language=dict(
                    freeze_encoder=True,
                    freeze_projection=True,
                ),
            ),
            agent_kwargs=dict(
                alignment=0.0,
            ),
        ),
        "sg_l_clip_tune": update_config(
            clip_resume_path=None,
            task_encoders=dict(
                image="clip_vision_with_projection",
                language="clip_text_with_projection",
            ),
            task_encoder_kwargs=dict(
                image=dict(),
                language=dict(),
            ),
            agent_kwargs=dict(
                alignment=0.0,
            ),
            dataset_kwargs=dict(
                clip_preprocessing=True,
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
        ),
        "sg_l_clip_frozen": update_config(
            clip_resume_path=None,
            task_encoders=dict(
                image="clip_vision_with_projection",
                language="clip_text_with_projection",
            ),
            task_encoder_kwargs=dict(
                image=dict(
                    freeze_encoder=True,
                    freeze_projection=True,
                ),
                language=dict(
                    freeze_encoder=True,
                    freeze_projection=True,
                ),
            ),
            agent_kwargs=dict(
                alignment=0.0,
            ),
            dataset_kwargs=dict(
                clip_preprocessing=True,
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
        ),
        "sg_l_clip_align_tune": update_config(
            clip_resume_path=None,
            task_encoders=dict(
                image="clip_vision_with_projection",
                language="clip_text_with_projection",
            ),
            task_encoder_kwargs=dict(
                image=dict(),
                language=dict(),
            ),
            agent_kwargs=dict(
                alignment=1.0,
            ),
            dataset_kwargs=dict(
                clip_preprocessing=True,
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
        ),
        "sz_clip_refuse": update_config(
            data_path=None,
            clip_resume_path=None,
            use_image_embeds_as_inputs=True,
            use_text_embeds_as_inputs=True,
            task_encoders=dict(
                image="resnetv1-18-bridge-task",
                language="resnetv1-18-bridge-task",
            ),
            task_encoder_kwargs=dict(
                image=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
                language=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
            ),
            agent_kwargs=dict(
                alignment=0.0,
            ),
            dataset_kwargs=dict(
                load_frozen_embeddings=True,
            ),
        ),
        "sg_l_align_resnet_muse": update_config(
            task_encoders=dict(
                image="resnetv1-18-bridge",
                language="muse",
            ),
            task_encoder_kwargs=dict(
                image=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
                language=dict(),
            ),
            agent_kwargs=dict(
                alignment=1.0,
            ),
        ),
        "sg_sl_align_resnet_muse": update_config(
            task_encoders=dict(
                image="resnetv1-18-bridge",
                language="resnetv1-18-bridge-task",
            ),
            task_encoder_kwargs=dict(
                image=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
                language=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
            ),
            agent_kwargs=dict(
                alignment=1.0,
            ),
        ),
        "sg_sl_resnet_muse": update_config(
            task_encoders=dict(
                image="resnetv1-18-bridge",
                language="resnetv1-18-bridge-task",
            ),
            task_encoder_kwargs=dict(
                image=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
                language=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
            ),
            agent_kwargs=dict(
                alignment=0.0,
            ),
        ),
        "bc_resnet_muse": update_config(
            task_encoders=dict(
                image="resnetv1-18-bridge",
                language="muse",
            ),
            task_encoder_kwargs=dict(
                image=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
                language=dict(),
            ),
            agent_kwargs=dict(
                alignment=0.0,
            ),
        ),
        "lcbc": update_config(
            data_path=None,
            task_encoders=dict(
                language="muse",
                language_low_level="muse",
                language_joint="muse",
            ),
            task_encoder_kwargs=dict(
                image=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
                language=dict(),
                language_low_level=dict(),
                language_joint=dict(),
            ),
            agent_kwargs=dict(
                alignment=0.0,
                early_fuse_initial_obs=True,
            ),
            num_annotations=None,
            augment_language=False,
        ),
        "lcbc_fixed": update_config(
            data_path=None,
            task_encoders=dict(
                language="muse",
                language_low_level="muse",
                language_joint="muse",
            ),
            task_encoder_kwargs=dict(
                image=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
                language=dict(),
                language_low_level=dict(),
                language_joint=dict(),
            ),
            agent_kwargs=dict(
                alignment=0.0,
                early_fuse_initial_obs=True,
            ),
            num_annotations=None,
            augment_language=False,
        ),
        "lcbc_noinitial": update_config(
            data_path=None,
            task_encoders=dict(
                language="muse",
            ),
            task_encoder_kwargs=dict(
                image=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
                language=dict(),
            ),
            agent_kwargs=dict(
                alignment=0.0,
            ),
            dataset_kwargs=dict(
                labeled_ony=True,
            ),
        ),
        "sg_sl_align_clip_lang_frozen_ss2": update_config(
            ss2_batch_size=128,
            clip_resume_path=None,
            task_encoders=dict(
                image="resnetv1-18-bridge",
                language="clip_text_with_projection",
            ),
            task_encoder_kwargs=dict(
                image=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                    mlp_kwargs=dict(
                        hidden_dims=(
                            128,
                            512,
                        ),
                        activation=nn.relu,
                    ),
                ),
                language=dict(
                    freeze_encoder=True,
                    freeze_projection=True,
                ),
            ),
            agent_kwargs=dict(
                alignment=1.0,
            ),
        ),
        "sg_sl_align_muse_frozen_ss2": update_config(
            ss2_batch_size=128,
            clip_resume_path=None,
            task_encoders=dict(
                image="resnetv1-18-bridge",
                language="muse",
            ),
            task_encoder_kwargs=dict(
                image=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                    mlp_kwargs=dict(
                        hidden_dims=(
                            128,
                            512,
                        ),
                        activation=nn.relu,
                    ),
                ),
                language=dict(),
            ),
            agent_kwargs=dict(
                alignment=1.0,
            ),
        ),
        "bc_clip_frozen_embeddings": update_config(
            data_path=None,
            encoder="resnetv1-34-bridge-task-x-units",
            encoder_kwargs=dict(
                pooling_method="avg",
                add_spatial_coordinates=True,
                act="swish",
                task_units=50,
            ),
            drop_encoders=True,
            use_image_embeddings=True,
            use_text_embeddings=True,
            task_encoders=dict(
                image="clip_vision_with_projection",
                language="clip_text_with_projection",
            ),
            task_encoder_kwargs=dict(
                image=dict(
                    freeze_encoder=True,
                ),
                language=dict(
                    freeze_encoder=True,
                    freeze_projection=True,
                ),
            ),
            agent_kwargs=dict(
                alignment=0.0,
            ),
            dataset_kwargs=dict(
                load_frozen_embeddings=True,
            ),
        ),
    }

    possible_structures["bc_r3m_frozen"] = update_config(
        possible_structures["bc_clip_frozen_embeddings"],
        data_path=None,
        task_encoders=dict(
            language="muse",
        ),
        use_image_embeds_as_inputs=True,
        use_text_embeddings=False,
        encoder_kwargs=dict(
            task_units=256,
        ),
        drop_encoders=False,
    )

    possible_structures["bc_clip_frozen_embeddings_64"] = update_config(
        possible_structures["bc_clip_frozen_embeddings"],
        encoder_kwargs=dict(
            pooling_method="avg",
            add_spatial_coordinates=True,
            act="swish",
            task_units=64,
        ),
    )

    possible_structures["bc_clip_frozen_embeddings_128"] = update_config(
        possible_structures["bc_clip_frozen_embeddings"],
        encoder_kwargs=dict(
            pooling_method="avg",
            add_spatial_coordinates=True,
            act="swish",
            task_units=128,
        ),
    )

    possible_structures["bc_clip_frozen_embeddings_256"] = update_config(
        possible_structures["bc_clip_frozen_embeddings"],
        encoder_kwargs=dict(
            pooling_method="avg",
            add_spatial_coordinates=True,
            act="swish",
            task_units=256,
        ),
    )

    possible_structures["bc_clip_frozen_embeddings_512"] = update_config(
        possible_structures["bc_clip_frozen_embeddings"],
        encoder_kwargs=dict(
            pooling_method="avg",
            add_spatial_coordinates=True,
            act="swish",
            task_units=512,
        ),
    )

    possible_structures["bc_clip_frozen_embeddings_512_resnet50_no_prefuse_mlp"] = (
        update_config(
            possible_structures["bc_clip_frozen_embeddings"],
            encoder="resnetv1-50-bridge",
            encoder_kwargs=dict(
                pooling_method="avg",
                add_spatial_coordinates=True,
                act="swish",
                task_units=512,
                pre_fuse_mlp=False,
            ),
        )
    )

    possible_structures["bc_clip_frozen_embeddings_256_resnet50"] = update_config(
        possible_structures["bc_clip_frozen_embeddings"],
        encoder="resnetv1-50-bridge",
        encoder_kwargs=dict(
            pooling_method="avg",
            add_spatial_coordinates=True,
            act="swish",
            task_units=256,
        ),
    )

    possible_structures["sg_sl_align_resnet_muse_ss2"] = update_config(
        possible_structures["sg_sl_align_resnet_muse"],
        ss2_batch_size=28,
    )

    possible_structures["sg_sl_align_resnet_clip"] = update_config(
        possible_structures["sg_sl_align_resnet_muse"],
        data_path=None,
        use_text_embeds_as_inputs=True,
    )

    possible_structures["sg_sl_align_resnet_muse_domain5050"] = update_config(
        possible_structures["sg_sl_align_resnet_muse"],
        domain_weight=0.5,
    )
    possible_structures["sg_sl_align_resnet_muse_domain9010"] = update_config(
        possible_structures["sg_sl_align_resnet_muse"],
        domain_weight=0.1,
    )

    possible_structures["sg_sl_align_resnet_muse_512"] = update_config(
        possible_structures["sg_sl_align_resnet_muse"],
        encoder_kwargs=dict(
            pooling_method="avg",
            add_spatial_coordinates=True,
            act="swish",
            task_units=512,
        ),
    )

    possible_structures["sg_sl_align_resnet_muse_128"] = update_config(
        possible_structures["sg_sl_align_resnet_muse"],
        encoder_kwargs=dict(
            pooling_method="avg",
            add_spatial_coordinates=True,
            act="swish",
            task_units=128,
        ),
    )

    possible_structures["sg_sl_align_resnet_muse_align10"] = update_config(
        possible_structures["sg_sl_align_resnet_muse"],
        agent_kwargs=dict(
            alignment=10.0,
        ),
    )
    possible_structures["sg_sl_align_resnet_muse_align100"] = update_config(
        possible_structures["sg_sl_align_resnet_muse"],
        agent_kwargs=dict(
            alignment=100.0,
        ),
    )
    possible_structures["sg_sl_align_resnet_muse_align0.3"] = update_config(
        possible_structures["sg_sl_align_resnet_muse"],
        agent_kwargs=dict(
            alignment=0.3,
        ),
    )

    possible_structures["g_sl_align_resnet_muse"] = update_config(
        possible_structures["sg_sl_align_resnet_muse"],
        task_encoder_kwargs=dict(
            image=dict(
                g_only=True,
            ),
        ),
    )

    possible_structures["sg_sl_align_resnet_muse_big"] = update_config(
        possible_structures["sg_sl_align_resnet_muse"],
        encoder="resnetv1-50-bridge",
        task_encoders=dict(
            image="resnetv1-34-bridge",
            language="resnetv1-34-bridge-task",
        ),
        encoder_kwargs=dict(
            pooling_method="avg",
            add_spatial_coordinates=True,
            act="swish",
            task_units=256,
        ),
    )

    possible_structures["sg_sl_align_resnet_raw_muse"] = update_config(
        possible_structures["sg_sl_align_resnet_muse"],
        task_encoder_kwargs=dict(
            language=dict(
                pre_fuse_mlp=False,
                task_units=512,
            ),
        ),
    )

    possible_structures["sz_align_clip_refuse"] = update_config(
        possible_structures["sz_clip_refuse"],
        agent_kwargs=dict(
            alignment=1.0,
        ),
    )

    possible_structures["sz_align_ftmap"] = update_config(
        possible_structures["sz_clip_refuse"],
        encoder="resnetv1-34-bridge",
        encoder_kwargs=dict(
            pooling_method="avg",
            add_spatial_coordinates=True,
            act="swish",
            task_units=256,
            fuse_ftmaps=True,
        ),
        task_encoders=dict(
            image="resnetv1-34-bridge-task",
            language="resnetv1-34-bridge-task",
        ),
        task_encoder_kwargs=dict(
            image=dict(
                return_ftmaps=True,
            ),
            language=dict(
                return_ftmaps=True,
            ),
        ),
        agent_kwargs=dict(
            alignment=1.0,
            metric="cosine",
            flatten_task_reps=True,
        ),
    )

    possible_structures["sz_align_ftmap_clip_l2_freezeB"] = update_config(
        possible_structures["sz_align_ftmap"],
        data_path=None,
        batch_size=64,
        task_encoders=dict(
            image="clip_vision_with_ftmap",
            language="clip_text_with_ftmap",
        ),
        use_image_embeds_as_inputs=False,
        use_text_embeds_as_inputs=False,
        use_image_embeddings=False,
        use_text_embeddings=False,
        task_encoder_kwargs=dict(
            image=dict(
                resnet_config="resnetv1-34-bridge-task",
                clip_kwargs=dict(
                    normalize=True,
                ),
                resnet_kwargs=dict(
                    return_ftmaps=True,
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                    task_units=256,
                ),
                _overwrite=True,
            ),
            language=dict(
                resnet_config="resnetv1-34-bridge-task",
                clip_kwargs=dict(
                    normalize=True,
                ),
                resnet_kwargs=dict(
                    return_ftmaps=True,
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                    task_units=256,
                ),
                _overwrite=True,
            ),
        ),
        agent_kwargs=dict(
            metric="mse",
            freeze_task_B=True,
            clip_encoder_lr_multiplier=0.1,
        ),
        dataset_kwargs=dict(
            clip_preprocessing=True,
            load_frozen_embeddings=False,
        ),
    )

    possible_structures["sz_align_ftmap_clip_cosine_freezeB"] = update_config(
        possible_structures["sz_align_ftmap_clip_l2_freezeB"],
        agent_kwargs=dict(
            metric="cosine",
        ),
    )

    possible_structures["sz_ftmap"] = update_config(
        possible_structures["sz_align_ftmap"],
        agent_kwargs=dict(
            alignment=0.0,
        ),
    )

    possible_structures["sz_align_ftmap_mse"] = update_config(
        possible_structures["sz_align_ftmap"],
        agent_kwargs=dict(
            alignment=1.0,
            metric="mse",
        ),
    )

    possible_structures["sz_share_encoders"] = update_config(
        possible_structures["sz_clip_refuse"],
        agent_kwargs=dict(
            alignment=0.0,
            share_encoders=True,
        ),
    )
    possible_structures["sz_share_encoders_ftmap"] = update_config(
        possible_structures["sz_align_ftmap"],
        agent_kwargs=dict(
            alignment=0.0,
            share_encoders=True,
        ),
    )

    possible_structures["sg_sl_align_muse_ftmap"] = update_config(
        possible_structures["sz_align_ftmap"],
        agent_kwargs=dict(
            alignment=1.0,
        ),
        task_encoders=dict(
            image="resnetv1-18-bridge",
        ),
        use_image_embeds_as_inputs=False,
        use_text_embeds_as_inputs=False,
        dataset_kwargs=dict(
            load_frozen_embeddings=False,
        ),
    )

    possible_structures["bc_clip_ss0_early_fusion"] = update_config(
        possible_structures["bc_clip_frozen_embeddings_256"],
        agent_kwargs=dict(
            early_fuse_initial_obs=True,
        ),
    )

    possible_structures["bc_clip_thawed_ss0_ef"] = update_config(
        clip_resume_path=None,
        encoder="resnetv1-34-bridge",
        encoder_kwargs=dict(
            pooling_method="avg",
            add_spatial_coordinates=True,
            act="swish",
            task_units=256,
        ),
        task_encoders=dict(
            image="clip_vision_with_projection",
            language="clip_text_with_projection",
        ),
        task_encoder_kwargs=dict(
            image=dict(
                normalize=True,
            ),
            language=dict(
                normalize=True,
            ),
        ),
        agent_kwargs=dict(
            alignment=0.0,
            early_fuse_initial_obs=True,
        ),
        dataset_kwargs=dict(
            clip_preprocessing=True,
        ),
    )

    possible_structures["bc_clip_thawed_ss0_ef_align1.0"] = update_config(
        possible_structures["bc_clip_thawed_ss0_ef"],
        agent_kwargs=dict(
            alignment=1.0,
        ),
    )

    possible_structures["bc_clip_thawed_ss0_ef_align0.1"] = update_config(
        possible_structures["bc_clip_thawed_ss0_ef"],
        agent_kwargs=dict(
            alignment=0.1,
        ),
    )

    possible_structures["bc_clip_thawed_ss0_ef_0.1x_lr"] = update_config(
        possible_structures["bc_clip_thawed_ss0_ef"],
        agent_kwargs=dict(
            clip_encoder_lr_multiplier=0.1,
        ),
    )

    possible_structures["bc_clip_thawed_ss0_ef_0.01x_lr"] = update_config(
        possible_structures["bc_clip_thawed_ss0_ef"],
        agent_kwargs=dict(
            clip_encoder_lr_multiplier=0.01,
        ),
    )

    possible_structures["bc_clip_thawed_ss0_ef_0.0x_lr"] = update_config(
        possible_structures["bc_clip_thawed_ss0_ef"],
        agent_kwargs=dict(
            clip_encoder_lr_multiplier=0.0,
        ),
    )

    possible_structures["bc_clip_frozen_ss0_ef"] = update_config(
        possible_structures["bc_clip_thawed_ss0_ef"],
        task_encoder_kwargs=dict(
            image=dict(
                freeze_encoder=True,
                freeze_projection=True,
            ),
            language=dict(
                freeze_encoder=True,
                freeze_projection=True,
            ),
        ),
    )

    possible_structures["bc_clip_align_0.1x_lr"] = update_config(
        possible_structures["bc_clip_thawed_ss0_ef"],
        agent_kwargs=dict(
            alignment=1.0,
            clip_encoder_lr_multiplier=0.1,
        ),
    )

    possible_structures["bc_clip_0.1x_lr"] = update_config(
        possible_structures["bc_clip_thawed_ss0_ef"],
        agent_kwargs=dict(
            alignment=0.0,
            clip_encoder_lr_multiplier=0.1,
        ),
    )

    possible_structures["bc_clip_align_0.01x_lr"] = update_config(
        possible_structures["bc_clip_thawed_ss0_ef"],
        agent_kwargs=dict(
            alignment=10.0,
            clip_encoder_lr_multiplier=0.01,
        ),
    )

    possible_structures["bc_clip_align_0.1x_goal_always_last"] = update_config(
        possible_structures["bc_clip_thawed_ss0_ef"],
        agent_kwargs=dict(
            alignment=1.0,
            clip_encoder_lr_multiplier=0.1,
        ),
        dataset_kwargs=dict(
            goal_relabel_last_proportion=1.0,
            language_keep_proportion=1.0,
        ),
    )

    possible_structures["bc_clip_align_1x_lr"] = update_config(
        possible_structures["bc_clip_thawed_ss0_ef"],
        agent_kwargs=dict(
            alignment=10.0,
            clip_encoder_lr_multiplier=1.0,
        ),
    )

    possible_structures["bc_clip_1x_lr"] = update_config(
        possible_structures["bc_clip_thawed_ss0_ef"],
        agent_kwargs=dict(
            alignment=0.0,
            clip_encoder_lr_multiplier=1.0,
        ),
    )

    possible_structures["bc_clip_1.0_align_0.1x_lr_frozen_lang"] = update_config(
        possible_structures["bc_clip_thawed_ss0_ef"],
        data_path=None,
        clip_resume_path=None,
        task_encoder_kwargs=dict(
            language=dict(
                freeze_encoder=True,
                freeze_projection=True,
            ),
        ),
        agent_kwargs=dict(
            alignment=1.0,
            clip_encoder_lr_multiplier=0.1,
            early_fuse_initial_obs=False,
        ),
        num_annotations=None,
    )

    possible_structures["lcbc_clip_lang"] = update_config(
        possible_structures["lcbc_noinitial"],
        task_encoders=dict(
            language="clip_text_with_projection",
        ),
        task_encoder_kwargs=dict(
            language=dict(
                freeze_encoder=True,
                freeze_projection=True,
            ),
        ),
        clip_resume_path=None,
    )

    possible_structures["bc_clip_1.0_align_0.1x_lr_thawed_lang"] = update_config(
        possible_structures["bc_clip_thawed_ss0_ef"],
        data_path=None,
        task_encoder_kwargs=dict(
            language=dict(
                freeze_encoder=False,
                freeze_projection=False,
            ),
        ),
        agent_kwargs=dict(
            alignment=1.0,
            clip_encoder_lr_multiplier=0.1,
            early_fuse_initial_obs=False,
        ),
    )

    possible_structures["bc_clip_1.0_align_0.1x_lr_frozen_img"] = update_config(
        possible_structures["bc_clip_thawed_ss0_ef"],
        data_path=None,
        task_encoder_kwargs=dict(
            image=dict(
                freeze_encoder=True,
                freeze_projection=True,
            ),
        ),
        agent_kwargs=dict(
            alignment=1.0,
            clip_encoder_lr_multiplier=0.1,
            early_fuse_initial_obs=False,
        ),
    )

    possible_structures["bc_clip_1.0_align_0.1x_lr_frozen_lang_image_from_lang"] = (
        update_config(
            possible_structures["bc_clip_1.0_align_0.1x_lr_frozen_lang"],
            agent_kwargs=dict(
                alignment=1.0,
                other_alignment=0.0,
            ),
        )
    )
    possible_structures["bc_clip_1.0_align_0.1x_lr_frozen_lang_image_from_lang"] = (
        update_config(
            possible_structures["bc_clip_1.0_align_0.1x_lr_frozen_lang"],
            agent_kwargs=dict(
                alignment=1.0,
                other_alignment=0.0,
            ),
        )
    )

    possible_structures["bc_clip_1.0_align_0.1x_lr_frozen_lang_lang_from_image"] = (
        update_config(
            possible_structures["bc_clip_1.0_align_0.1x_lr_frozen_lang"],
            agent_kwargs=dict(
                alignment=0.0,
                other_alignment=1.0,
            ),
        )
    )

    possible_structures["bc_clip_1.0_align_0.1x_lr_frozen_lang_freeze_B"] = (
        update_config(
            possible_structures["bc_clip_1.0_align_0.1x_lr_frozen_lang"],
            agent_kwargs=dict(
                freeze_task_B=True,
            ),
        )
    )

    possible_structures["bc_clip_1.0_align_0.1x_lr_thawed_lang_freeze_B"] = (
        update_config(
            possible_structures["bc_clip_1.0_align_0.1x_lr_thawed_lang"],
            agent_kwargs=dict(
                freeze_task_B=True,
            ),
        )
    )

    possible_structures["bc_clip_1.0_align_0.1x_lr_tune_proj"] = update_config(
        possible_structures["bc_clip_thawed_ss0_ef"],
        data_path=None,
        task_encoder_kwargs=dict(
            language=dict(
                freeze_encoder=True,
                freeze_projection=False,
            ),
            image=dict(
                freeze_encoder=True,
                freeze_projection=False,
            ),
        ),
        agent_kwargs=dict(
            alignment=1.0,
            clip_encoder_lr_multiplier=0.1,
            early_fuse_initial_obs=False,
        ),
    )

    possible_structures["bc_clip_GL"] = update_config(
        possible_structures["bc_clip_thawed_ss0_ef"],
        data_path=None,
        clip_resume_path=None,
        task_encoder_kwargs=dict(
            language=dict(
                freeze_encoder=True,
                freeze_projection=True,
            ),
        ),
        agent_kwargs=dict(
            alignment=1.0,
            clip_encoder_lr_multiplier=0.1,
            early_fuse_initial_obs=False,
            no_initial=True,
        ),
    )

    possible_structures["bc_clip_no_align_0.1x_lr_frozen_lang"] = update_config(
        possible_structures["bc_clip_1.0_align_0.1x_lr_frozen_lang"],
        data_path=None,
        clip_resume_path=None,
        agent_kwargs=dict(
            alignment=0.0,
        ),
    )

    possible_structures["bc_clip_prealign_0.1x_lr_frozen_lang"] = update_config(
        possible_structures["bc_clip_1.0_align_0.1x_lr_frozen_lang"],
        data_path=None,
        agent_kwargs=dict(
            alignment=0.0,
        ),
    )

    possible_structures["bc_clip_frozen"] = update_config(
        possible_structures["bc_clip_1.0_align_0.1x_lr_frozen_lang"],
        data_path=None,
        task_encoder_kwargs=dict(
            image=dict(
                freeze_encoder=True,
                freeze_projection=True,
            ),
            language=dict(
                freeze_encoder=True,
                freeze_projection=True,
            ),
        ),
    )

    possible_structures["bc_clip_frozen_rebuttal_1k"] = update_config(
        possible_structures["bc_clip_1.0_align_0.1x_lr_frozen_lang"],
        clip_resume_path=None,
        data_path=None,
        num_annotations=1000,
        task_encoder_kwargs=dict(
            image=dict(
                freeze_encoder=True,
                freeze_projection=True,
            ),
            language=dict(
                freeze_encoder=True,
                freeze_projection=True,
            ),
        ),
    )

    possible_structures["bc_clip_frozen_rebuttal_1k"] = update_config(
        possible_structures["bc_clip_1.0_align_0.1x_lr_frozen_lang"],
        clip_resume_path=None,
        data_path=None,
        num_annotations=1000,
        task_encoder_kwargs=dict(
            image=dict(
                freeze_encoder=True,
                freeze_projection=True,
            ),
            language=dict(
                freeze_encoder=True,
                freeze_projection=True,
            ),
        ),
    )

    possible_structures["bc_clip_frozen_rebuttal_3k"] = update_config(
        possible_structures["bc_clip_1.0_align_0.1x_lr_frozen_lang"],
        clip_resume_path=None,
        data_path=None,
        num_annotations=3000,
        task_encoder_kwargs=dict(
            image=dict(
                freeze_encoder=True,
                freeze_projection=True,
            ),
            language=dict(
                freeze_encoder=True,
                freeze_projection=True,
            ),
        ),
    )

    possible_structures["bc_clip_frozen_rebuttal_5k"] = update_config(
        possible_structures["bc_clip_1.0_align_0.1x_lr_frozen_lang"],
        clip_resume_path=None,
        data_path=None,
        num_annotations=5000,
        task_encoder_kwargs=dict(
            image=dict(
                freeze_encoder=True,
                freeze_projection=True,
            ),
            language=dict(
                freeze_encoder=True,
                freeze_projection=True,
            ),
        ),
    )

    possible_structures["bc_clip_frozen_rebuttal_7k"] = update_config(
        possible_structures["bc_clip_1.0_align_0.1x_lr_frozen_lang"],
        clip_resume_path=None,
        data_path=None,
        num_annotations=7000,
        task_encoder_kwargs=dict(
            image=dict(
                freeze_encoder=True,
                freeze_projection=True,
            ),
            language=dict(
                freeze_encoder=True,
                freeze_projection=True,
            ),
        ),
    )

    possible_structures["bc_clip_frozen_rebuttal_aug"] = update_config(
        possible_structures["bc_clip_1.0_align_0.1x_lr_frozen_lang"],
        clip_resume_path=None,
        data_path=None,
        task_encoder_kwargs=dict(
            image=dict(
                freeze_encoder=True,
                freeze_projection=True,
            ),
            language=dict(
                freeze_encoder=True,
                freeze_projection=True,
            ),
        ),
    )

    possible_structures["bc_clip_frozen_rebuttal_no_aug"] = update_config(
        possible_structures["bc_clip_1.0_align_0.1x_lr_frozen_lang"],
        clip_resume_path=None,
        data_path=None,
        augment_language=False,
        task_encoder_kwargs=dict(
            image=dict(
                freeze_encoder=True,
                freeze_projection=True,
            ),
            language=dict(
                freeze_encoder=True,
                freeze_projection=True,
            ),
        ),
    )

    possible_structures["bcz"] = update_config(
        possible_structures["sg_l_align_resnet_muse"],
        data_path=None,
    )

    possible_structures["bcz_fixed"] = update_config(
        possible_structures["sg_l_align_resnet_muse"],
        data_path=None,
        agent_kwargs=dict(
            alignment=1.0,
            metric="cosine_loss",
        ),
    )

    possible_structures["bcz_fixed_rebuttal_rnet18"] = update_config(
        possible_structures["sg_l_align_resnet_muse"],
        data_path=None,
        task_encoders=dict(
            image="resnetv1-18-bridge",
        ),
        agent_kwargs=dict(
            alignment=1.0,
            metric="cosine_loss",
        ),
    )

    possible_structures["bcz_fixed_rebuttal_rnet50"] = update_config(
        possible_structures["sg_l_align_resnet_muse"],
        data_path=None,
        task_encoders=dict(
            image="resnetv1-50-bridge",
        ),
        agent_kwargs=dict(
            alignment=1.0,
            metric="cosine_loss",
        ),
    )

    possible_structures["language_play"] = update_config(
        possible_structures["sg_sl_align_resnet_muse"],
        data_path=None,
        agent_kwargs=dict(
            alignment=0.0,
        ),
    )

    possible_structures["bc_clip_from_scratch"] = update_config(
        possible_structures["bc_clip_1.0_align_0.1x_lr_frozen_lang"],
        data_path=None,
        clip_resume_path=None,
    )

    possible_structures["bc_clip_no_prealign"] = update_config(
        possible_structures["bc_clip_1.0_align_0.1x_lr_frozen_lang"],
        data_path=None,
    )

    possible_structures["bc_clip_no_prealign_fixed"] = update_config(
        possible_structures["bc_clip_1.0_align_0.1x_lr_frozen_lang"],
        data_path=None,
        clip_resume_path=None,
        task_encoder_kwargs=dict(
            language=dict(
                freeze_encoder=False,
                freeze_projection=False,
            ),
            agent_kwargs=dict(
                alignment=0.1,
                clip_encoder_lr_multiplier=0.1,
            ),
        ),
    )

    possible_structures["bc_clip_0.1_align_1.0x_lr_frozen_lang"] = update_config(
        possible_structures["bc_clip_thawed_ss0_ef"],
        task_encoder_kwargs=dict(
            language=dict(
                freeze_encoder=True,
                freeze_projection=True,
            ),
        ),
        agent_kwargs=dict(
            alignment=0.1,
            clip_encoder_lr_multiplier=1.0,
            early_fuse_initial_obs=False,
        ),
    )

    possible_structures["bc_clip_ss2_bridge_embeds_frozen"] = update_config(
        possible_structures["bc_clip_thawed_ss0_ef"],
        clip_resume_path=None,
        task_encoder_kwargs=dict(
            image=dict(
                freeze_encoder=True,
                freeze_projection=True,
            ),
            language=dict(
                freeze_encoder=True,
                freeze_projection=True,
            ),
        ),
        agent_kwargs=dict(
            alignment=0.0,
            clip_encoder_lr_multiplier=0.1,
            early_fuse_initial_obs=False,
        ),
    )

    possible_structures["bc_clip_ls0_align"] = update_config(
        possible_structures["sz_clip_refuse"],
        use_image_embeds_as_inputs=False,
        use_image_embeddings=True,
        agent_kwargs=dict(
            alignment=1.0,
        ),
    )

    possible_structures["bc_clip_ls0"] = update_config(
        possible_structures["sz_clip_refuse"],
        use_image_embeds_as_inputs=False,
        use_image_embeddings=True,
        agent_kwargs=dict(
            alignment=0.0,
        ),
    )

    possible_structures["bc_clip_ls0_residual_align"] = update_config(
        possible_structures["sz_clip_refuse"],
        use_image_embeds_as_inputs=False,
        use_image_embeddings=True,
        task_encoder_kwargs=dict(
            language=dict(
                task_highway=True,
                normalize_output=True,
            ),
        ),
        agent_kwargs=dict(
            alignment=1.0,
        ),
    )

    possible_structures["bc_clip_ls0_residual_align_cos"] = update_config(
        possible_structures["bc_clip_ls0_residual_align"],
        agent_kwargs=dict(
            alignment=1.0,
            metric="cosine_loss",
        ),
    )

    return possible_structures[config_string]
