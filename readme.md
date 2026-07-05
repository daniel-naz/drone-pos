0. py -m pip install yt-dlp

1. py .\download_vids.py

1. python -m pip install opencv-python numpy pillow

2. python .\dataset\extract_frames.py `
  -i ".\unprocessed\videos" `
  -f 60 `
  -o ".\dataset\frames"

3. python .\dataset\remove_similar.py `
  -i ".\dataset\frames" `
  -p 97

4. python .\dataset\build_match_graph.py `
  -i ".\dataset\frames" `
  -db ".\dataset\graph.db" `
  --max-size 1000 `
  --max-features 2500 `
  --same-folder-window 30 `
  --cross-folder-top-k 10 `
  --workers 6 `
  --matcher flann `
  --overwrite

5. python .\dataset\analyze_transforms.py `
  -db ".\dataset\graph.db" `
  -tp ".\unprocessed\SRT" `
  --base-fov 60 `
  --use-dzoom

6. py .\dataset\video_position_map_server_current_frame.py `
  -v ".\unprocessed\videos\DJI_0149.mp4" `
  -db ".\dataset\graph.db" `
  -r "." `
  --global-estimator-script ".\dataset\estimate_image_position.py" `
  --bfs-estimator-script ".\dataset\estimate_image_position_bfs_only.py" `
  --global-candidate-steps "0" `
  --global-timeout 45 `
  --local-timeout 8 `
  --needed-matches 1 `
  --bfs-depth 2 `
  --bfs-neighbor-limit 40 `
  --bfs-max-candidates 200 `
  --feature-max-size 1000 `
  --max-features 1000 `
  --ratio 0.75 `
  --ransac 5 `
  --min-good 12 `
  --min-inliers 8 `
  --min-inlier-ratio 0.15 `
  --min-coverage 0.01 `
  --min-confidence 0.08 `
  --max-reprojection-error 15 `
  --workers 6 `
  --auto-every 2
