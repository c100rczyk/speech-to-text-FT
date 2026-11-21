# get each file from experiments/ and run for them one training

set -u
for cfg in configs/experiments/*.yaml; do
  name=$(basename "$cfg" .yaml) # cut folder and extension.
  out="runs/$name" # Remains only core "experiments/01_small_r16.yaml" -> "01_small_r16"
  [ -f "$out/DONE" ] && continue  # If in results folder is DONE file, dont run this training
  mkdir -p "$out" # Create folder if not exists

  if python train.py --config "$cfg" --output_dir "$out" > "$out/train.log" 2>&1

  then touch "$out/DONE"  #Create empty file "DONE" after success
  else echo "$name" >> runs/failed.txt
  fi

done