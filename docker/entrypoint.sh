#!/bin/bash
# Run as the host user so files written to the bind-mounted repo
# (work_dirs/, data/*.json, ...) are owned by the host uid:gid.

if [ -n "${LOCAL_USER_ID}" ]; then
	echo "Starting with UID : ${LOCAL_USER_ID} GID : ${LOCAL_GROUP_ID:-unchanged}"
	if [ -n "${LOCAL_GROUP_ID}" ]; then
		groupmod -o -g "${LOCAL_GROUP_ID}" user
	fi
	usermod -o -u "${LOCAL_USER_ID}" user
	chown "${LOCAL_USER_ID}:${LOCAL_GROUP_ID:-user}" /home/user
	exec gosu user "$@"
else
	exec "$@"
fi
