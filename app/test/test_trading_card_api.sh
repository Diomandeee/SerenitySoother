#!/bin/bash

# Base URL
BASE_URL="http://localhost:8000"

# Path to the logo and background images
LOGO_PATH="images/logo.PNG"
BACKGROUND_IMAGE_PATH="images/file-uFeqtwb8Bw3nbxivPmMb4tiY.png"

# Function to create a trading card
create_trading_card() {
  echo "Creating a new trading card..."
  curl -s -X POST "$BASE_URL/trading_cards/" -H "Content-Type: application/json" -d '{
    "user_id": 1,
    "card_type": "meditation",
    "card_design": "serene",
    "realm_access_url": "http://example.com/realm",
    "qr_code_url": "'"$(curl -s -X POST "$BASE_URL/generate_qr_code" -H "Content-Type: application/json" -d '{ "data": "http://example.com/realm", "box_size": 10, "border": 4, "fill_color": "blue", "back_color": "white", "logo_path": "'"$LOGO_PATH"'", "background_image_path": "'"$BACKGROUND_IMAGE_PATH"'" }' | jq -r '.qr_code_url')"'" 
  }'
  echo -e "\n"
}

# Function to get all trading cards
get_trading_cards() {
  echo "Getting all trading cards..."
  curl -s -X GET "$BASE_URL/trading_cards/"
  echo -e "\n"
}

# Function to get a trading card by ID
get_trading_card() {
  local trading_card_id=$1
  echo "Getting trading card with ID $trading_card_id..."
  curl -s -X GET "$BASE_URL/trading_cards/$trading_card_id"
  echo -e "\n"
}

# Function to update a trading card
update_trading_card() {
  local trading_card_id=$1
  echo "Updating trading card with ID $trading_card_id..."
  curl -s -X PUT "$BASE_URL/trading_cards/$trading_card_id" -H "Content-Type: application/json" -d '{
    "card_design": "updated design"
  }'
  echo -e "\n"
}

# Function to delete a trading card
delete_trading_card() {
  local trading_card_id=$1
  echo "Deleting trading card with ID $trading_card_id..."
  curl -s -X DELETE "$BASE_URL/trading_cards/$trading_card_id"
  echo -e "\n"
}

# Run the tests
create_trading_card
get_trading_cards
get_trading_card 1
update_trading_card 1
delete_trading_card 1
get_trading_cards

echo "API Testing Completed"


chmod +x test_trading_card_api.sh
