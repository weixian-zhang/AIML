#!/bin/bash

set -e

# Set the script folder directory for navigation
script_dir="$(dirname -- "${BASH_SOURCE[0]}")"

SUBSCRIPTION=${1:-$SUBSCRIPTION}
if [ -z "$SUBSCRIPTION" ]; then
  read -r -p "Please enter a Subscription ID: " SUBSCRIPTION
fi

RESOURCE_GROUP=${2:-$RESOURCE_GROUP}
if [ -z "$RESOURCE_GROUP" ]; then
  read -r -p "Enter Resource Group: " RESOURCE_GROUP
fi

ALERT_RG=${3:-$ALERT_RG}
if [ -z "$ALERT_RG" ]; then
  read -r -p "Please enter a Resource Group where you would want to deploy the alerts: " ALERT_RG
  if [ "$(az group exists --subscription "$SUBSCRIPTION" --name "$ALERT_RG")" = false ]; then
    echo "Resource Group '$ALERT_RG' not present in subscription '$SUBSCRIPTION'. Exiting."
    exit 1
  fi
fi

ACTION_GROUP_IDS=${4-$ACTION_GROUP_IDS}

alertScope="/subscriptions/$SUBSCRIPTION"

for alert in "$script_dir"/activityLogAlerts/*.json; do
  echo "Creating activity log alert from: ${alert}"
  az deployment group create --no-prompt --no-wait \
    --subscription "$SUBSCRIPTION" \
    --name "$(basename "${alert}" .json)_alert" \
    --resource-group "$ALERT_RG" \
    --template-file "$script_dir/templates/activityLogAlerts.bicep" \
    --parameters @"$alert" resourceGroup="$RESOURCE_GROUP" alertScope="$alertScope" actionGroupIds="$ACTION_GROUP_IDS"
done
for alert in "$script_dir"/activityLogAlerts/*.json; do
  az deployment group wait --created \
    --subscription "$SUBSCRIPTION" \
    --name "$(basename "${alert}" .json)_alert" \
    --resource-group "$ALERT_RG" \
    --interval 10 --timeout 120
done
