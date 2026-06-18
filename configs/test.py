# ======================================================
# GeoX Inference Configuration
# ======================================================

# == Model Configuration ==
model_url = "./checkpoints/model.safetensors"
model_load_strict = False

# == Inference Configuration ==
target_size = 518  # Target size for input images

# == Visualization Configuration ==
vis_conf_threshold = 0.2  # Initial percentage of low-confidence points to filter out
vis_filter_by_frames = "All"
vis_mask_black_bg = False
vis_mask_white_bg = False
vis_mask_sky = False
vis_show_cam = True
vis_prediction_mode = "Predicted Depth"
vis_use_point_map = False  # Use point map instead of depth-based points
vis_background_mode = False  # Run viser server in background mode
vis_port = 8082  # Port number for the viser server

# == Export Configuration ==
save_glb = False  # Save the output as a GLB file



