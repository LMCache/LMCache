ffmpeg -i in.mp4 -an \
  -vf "fps=2" \
  -c:v libx264 -profile:v baseline \
  -bf 0 -refs 1 \
  -g 16 -keyint_min 16 -sc_threshold 0 \
  -x264-params "open-gop=0" \
  out.mp4