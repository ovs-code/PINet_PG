for model in ./checkpoints/*
do
    echo -n "Creating report for $model ..."
    if python report.py $model &> /dev/null ; then
        echo " done."
    else
        echo " failed."
    fi
done