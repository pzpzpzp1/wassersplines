render:
	~/blender-3.0.0-linux-x64/blender --background --python render/render_mballs.py  -- --cycles-device CUDA --out rings_rig --trajs rings_rig.npy --scene render/rings.blend --smooth_factor 2 --smooth_iters 50
