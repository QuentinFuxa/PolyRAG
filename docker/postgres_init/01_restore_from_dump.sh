#!/bin/bash
set -e

BACKUP_FILE_IN_CONTAINER="/docker-entrypoint-initdb.d/backup.sql"

echo "PostgreSQL init script started."
echo "Looking for backup file at: $BACKUP_FILE_IN_CONTAINER"
echo "Contents of /docker-entrypoint-initdb.d/:"
ls -la /docker-entrypoint-initdb.d/

# Check if the backup file exists at the expected path in the container
if [ -f "$BACKUP_FILE_IN_CONTAINER" ]; then
  echo "Attempting to restore database from $BACKUP_FILE_IN_CONTAINER..."
  # POSTGRES_USER and POSTGRES_DB should be available from .env via compose.yaml
  if [ -z "$POSTGRES_USER" ] || [ -z "$POSTGRES_DB" ]; then
    echo "Error: POSTGRES_USER or POSTGRES_DB environment variables are not set."
    exit 1
  fi
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" < "$BACKUP_FILE_IN_CONTAINER"
  echo "Database restore complete from $BACKUP_FILE_IN_CONTAINER."
else
  echo "Backup file $BACKUP_FILE_IN_CONTAINER not found. Skipping database restore."
  echo "Host SQL_DUMP_PATH was: $SQL_DUMP_PATH (for reference, check if this path is correct on the host and if the file exists there)."
fi
echo "PostgreSQL init script finished."
