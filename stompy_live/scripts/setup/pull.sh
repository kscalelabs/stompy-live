#!/bin/bash

# Fetch the latest changes from the remote repository
git fetch

# Check for differences between the local and remote branches
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse @{u})

if [ "$LOCAL" = "$REMOTE" ]; then
	echo "YES"
else
	git pull
	sudo systemctl restart stompy-live
fi
