MODEL_CONFIG=$1
SLIM_CONFIG=$2

if [ -e "slim_scripts" ]; then
    rm slim_scripts
else
    touch slim_scripts
fi

echo python tools/train.py -c $MODEL_CONFIG --slim_config $SLIM_CONFIG >> slim_scripts
echo >> slim_scripts
echo python tools/eval.py -c $MODEL_CONFIG --slim_config $SLIM_CONFIG \ >> slim_scripts
echo    -o weights=output/$SLIM_CONFIG/model_final >> slim_scripts
echo >> slim_scripts
echo python tools/infer.py -c $MODEL_CONFIG --slim_config $SLIM_CONFIG \ >> slim_scripts
echo     -o weights=output/!!!/model_final >> slim_scripts
echo     --infer_img=IMAGE_PATH >> slim_scripts
echo >> slim_scripts
echo python tools/export_model.py -c $MODEL_CONFIG --slim_config $SLIM_CONFIG \ >> slim_scripts
echo     -o weights=output/!!!/model_final >> slim_scripts