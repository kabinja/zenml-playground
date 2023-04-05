#!/bin/bash

CURRENT_STACK=$(zenml stack get | grep "The repository active stack is" | cut -d\' -f2)
if [ $# -eq 0 ]
    then
        STACK=$CURRENT_STACK
    else
        STACK=$1
fi

if [[ "$CURRENT_STACK" != "$STACK" ]]
    then
        zenml stack down
        zenml down
        zenml stack set $STACK
        zenml up
        zenml stack up
fi

zenml stack describe
python "$(dirname $0)/src/run.py"