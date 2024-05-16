#!/bin/bash

# Base URL
BASE_URL="http://localhost:8000"

# Function to create a user
create_user() {
  echo "Creating a new user..."
  curl -s -X POST "$BASE_URL/users/" -H "Content-Type: application/json" -d '{
    "username": "testuser",
    "email": "testuser@example.com",
    "password": "password123"
  }'
  echo -e "\n"
}

# Function to get all users
get_users() {
  echo "Getting all users..."
  curl -s -X GET "$BASE_URL/users/"
  echo -e "\n"
}

# Function to get a user by ID
get_user() {
  local user_id=$1
  echo "Getting user with ID $user_id..."
  curl -s -X GET "$BASE_URL/users/$user_id"
  echo -e "\n"
}

# Function to update a user
update_user() {
  local user_id=$1
  echo "Updating user with ID $user_id..."
  curl -s -X PUT "$BASE_URL/users/$user_id" -H "Content-Type: application/json" -d '{
    "username": "updateduser",
    "email": "updateduser@example.com"
  }'
  echo -e "\n"
}

# Function to delete a user
delete_user() {
  local user_id=$1
  echo "Deleting user with ID $user_id..."
  curl -s -X DELETE "$BASE_URL/users/$user_id"
  echo -e "\n"
}

# Function to create a session
create_session() {
  echo "Creating a new session..."
  curl -s -X POST "$BASE_URL/sessions/" -H "Content-Type: application/json" -d '{
    "user_id": 1,
    "session_type": "meditation",
    "session_status": "active"
  }'
  echo -e "\n"
}

# Function to get all sessions
get_sessions() {
  echo "Getting all sessions..."
  curl -s -X GET "$BASE_URL/sessions/"
  echo -e "\n"
}

# Function to get a session by ID
get_session() {
  local session_id=$1
  echo "Getting session with ID $session_id..."
  curl -s -X GET "$BASE_URL/sessions/$session_id"
  echo -e "\n"
}

# Function to update a session
update_session() {
  local session_id=$1
  echo "Updating session with ID $session_id..."
  curl -s -X PUT "$BASE_URL/sessions/$session_id" -H "Content-Type: application/json" -d '{
    "session_status": "completed"
  }'
  echo -e "\n"
}

# Function to delete a session
delete_session() {
  local session_id=$1
  echo "Deleting session with ID $session_id..."
  curl -s -X DELETE "$BASE_URL/sessions/$session_id"
  echo -e "\n"
}

# Function to create a script
create_script() {
  echo "Creating a new script..."
  curl -s -X POST "$BASE_URL/scripts/" -H "Content-Type: application/json" -d '{
    "session_id": 1,
    "script_type": "guided",
    "script_content": "This is a test script"
  }'
  echo -e "\n"
}

# Function to get all scripts
get_scripts() {
  echo "Getting all scripts..."
  curl -s -X GET "$BASE_URL/scripts/"
  echo -e "\n"
}

# Function to get a script by ID
get_script() {
  local script_id=$1
  echo "Getting script with ID $script_id..."
  curl -s -X GET "$BASE_URL/scripts/$script_id"
  echo -e "\n"
}

# Function to update a script
update_script() {
  local script_id=$1
  echo "Updating script with ID $script_id..."
  curl -s -X PUT "$BASE_URL/scripts/$script_id" -H "Content-Type: application/json" -d '{
    "script_content": "Updated script content"
  }'
  echo -e "\n"
}

# Function to delete a script
delete_script() {
  local script_id=$1
  echo "Deleting script with ID $script_id..."
  curl -s -X DELETE "$BASE_URL/scripts/$script_id"
  echo -e "\n"
}

# Function to create a scene
create_scene() {
  echo "Creating a new scene..."
  curl -s -X POST "$BASE_URL/scenes/" -H "Content-Type: application/json" -d '{
    "script_id": 1,
    "scene_type": "introduction",
    "scene_description": "This is the introduction scene"
  }'
  echo -e "\n"
}

# Function to get all scenes
get_scenes() {
  echo "Getting all scenes..."
  curl -s -X GET "$BASE_URL/scenes/"
  echo -e "\n"
}

# Function to get a scene by ID
get_scene() {
  local scene_id=$1
  echo "Getting scene with ID $scene_id..."
  curl -s -X GET "$BASE_URL/scenes/$scene_id"
  echo -e "\n"
}

# Function to update a scene
update_scene() {
  local scene_id=$1
  echo "Updating scene with ID $scene_id..."
  curl -s -X PUT "$BASE_URL/scenes/$scene_id" -H "Content-Type: application/json" -d '{
    "scene_description": "Updated scene description"
  }'
  echo -e "\n"
}

# Function to delete a scene
delete_scene() {
  local scene_id=$1
  echo "Deleting scene with ID $scene_id..."
  curl -s -X DELETE "$BASE_URL/scenes/$scene_id"
  echo -e "\n"
}

# Function to create an element
create_element() {
  echo "Creating a new element..."
  curl -s -X POST "$BASE_URL/elements/" -H "Content-Type: application/json" -d '{
    "scene_id": 1,
    "element_type": "text",
    "element_description": "This is an element description"
  }'
  echo -e "\n"
}

# Function to get all elements
get_elements() {
  echo "Getting all elements..."
  curl -s -X GET "$BASE_URL/elements/"
  echo -e "\n"
}

# Function to get an element by ID
get_element() {
  local element_id=$1
  echo "Getting element with ID $element_id..."
  curl -s -X GET "$BASE_URL/elements/$element_id"
  echo -e "\n"
}

# Function to update an element
update_element() {
  local element_id=$1
  echo "Updating element with ID $element_id..."
  curl -s -X PUT "$BASE_URL/elements/$element_id" -H "Content-Type: application/json" -d '{
    "element_description": "Updated element description"
  }'
  echo -e "\n"
}

# Function to delete an element
delete_element() {
  local element_id=$1
  echo "Deleting element with ID $element_id..."
  curl -s -X DELETE "$BASE_URL/elements/$element_id"
  echo -e "\n"
}

# Function to create an emotion
create_emotion() {
  echo "Creating a new emotion..."
  curl -s -X POST "$BASE_URL/emotions/" -H "Content-Type: application/json" -d '{
    "user_id": 1,
    "emotion_type": "happy",
    "emotion_intensity": "high"
  }'
  echo -e "\n"
}

# Function to get all emotions
get_emotions() {
  echo "Getting all emotions..."
  curl -s -X GET "$BASE_URL/emotions/"
  echo -e "\n"
}

# Function to get an emotion by ID
get_emotion() {
  local emotion_id=$1
  echo "Getting emotion with ID $emotion_id..."
  curl -s -X GET "$BASE_URL/emotions/$emotion_id"
  echo -e "\n"
}

# Function to update an emotion
update_emotion() {
  local emotion_id=$1
  echo "Updating emotion with ID $emotion_id..."
  curl -s -X PUT "$BASE_URL/emotions/$emotion_id" -H "Content-Type: application/json" -d '{
    "emotion_description": "Updated emotion description"
  }'
  echo -e "\n"
}

# Function to delete an emotion
delete_emotion() {
  local emotion_id=$1
  echo "Deleting emotion with ID $emotion_id..."
  curl -s -X DELETE "$BASE_URL/emotions/$emotion_id"
  echo -e "\n"
}

# Function to create a goal
create_goal() {
  echo "Creating a new goal..."
  curl -s -X POST "$BASE_URL/goals/" -H "Content-Type: application/json" -d '{
    "user_id": 1,
    "goal_type": "meditation",
    "goal_status": "active"
  }'
  echo -e "\n"
}

# Function to get all goals
get_goals() {
  echo "Getting all goals..."
  curl -s -X GET "$BASE_URL/goals/"
  echo -e "\n"
}

# Function to get a goal by ID
get_goal() {
  local goal_id=$1
  echo "Getting goal with ID $goal_id..."
  curl -s -X GET "$BASE_URL/goals/$goal_id"
  echo -e "\n"
}

# Function to update a goal
update_goal() {
  local goal_id=$1
  echo "Updating goal with ID $goal_id..."
  curl -s -X PUT "$BASE_URL/goals/$goal_id" -H "Content-Type: application/json" -d '{
    "goal_description": "Updated goal description"
  }'
  echo -e "\n"
}

# Function to delete a goal
delete_goal() {
  local goal_id=$1
  echo "Deleting goal with ID $goal_id..."
  curl -s -X DELETE "$BASE_URL/goals/$goal_id"
  echo -e "\n"
}

# Function to create progress
create_progress() {
  echo "Creating a new progress..."
  curl -s -X POST "$BASE_URL/progress/" -H "Content-Type: application/json" -d '{
    "user_id": 1,
    "goal_id": 1,
    "progress_status": "in progress"
  }'
  echo -e "\n"
}

# Function to get all progress
get_all_progress() {
  echo "Getting all progress..."
  curl -s -X GET "$BASE_URL/progress/"
  echo -e "\n"
}

# Function to get progress by ID
get_progress() {
  local progress_id=$1
  echo "Getting progress with ID $progress_id..."
  curl -s -X GET "$BASE_URL/progress/$progress_id"
  echo -e "\n"
}

# Function to update progress
update_progress() {
  local progress_id=$1
  echo "Updating progress with ID $progress_id..."
  curl -s -X PUT "$BASE_URL/progress/$progress_id" -H "Content-Type: application/json" -d '{
    "progress_description": "Updated progress description"
  }'
  echo -e "\n"
}

# Function to delete progress
delete_progress() {
  local progress_id=$1
  echo "Deleting progress with ID $progress_id..."
  curl -s -X DELETE "$BASE_URL/progress/$progress_id"
  echo -e "\n"
}

# Function to create a notification
create_notification() {
  echo "Creating a new notification..."
  curl -s -X POST "$BASE_URL/notifications/" -H "Content-Type: application/json" -d '{
    "user_id": 1,
    "notification_type": "reminder"
  }'
  echo -e "\n"
}

# Function to get all notifications
get_notifications() {
  echo "Getting all notifications..."
  curl -s -X GET "$BASE_URL/notifications/"
  echo -e "\n"
}

# Function to get a notification by ID
get_notification() {
  local notification_id=$1
  echo "Getting notification with ID $notification_id..."
  curl -s -X GET "$BASE_URL/notifications/$notification_id"
  echo -e "\n"
}

# Function to update a notification
update_notification() {
  local notification_id=$1
  echo "Updating notification with ID $notification_id..."
  curl -s -X PUT "$BASE_URL/notifications/$notification_id" -H "Content-Type: application/json" -d '{
    "notification_message": "Updated notification message"
  }'
  echo -e "\n"
}

# Function to delete a notification
delete_notification() {
  local notification_id=$1
  echo "Deleting notification with ID $notification_id..."
  curl -s -X DELETE "$BASE_URL/notifications/$notification_id"
  echo -e "\n"
}

# Function to create a thought
create_thought() {
  echo "Creating a new thought..."
  curl -s -X POST "$BASE_URL/thoughts/" -H "Content-Type: application/json" -d '{
    "user_id": 1,
    "thought_type": "positive",
    "thought_description": "This is a positive thought"
  }'
  echo -e "\n"
}

# Function to get all thoughts
get_thoughts() {
  echo "Getting all thoughts..."
  curl -s -X GET "$BASE_URL/thoughts/"
  echo -e "\n"
}

# Function to get a thought by ID
get_thought() {
  local thought_id=$1
  echo "Getting thought with ID $thought_id..."
  curl -s -X GET "$BASE_URL/thoughts/$thought_id"
  echo -e "\n"
}

# Function to update a thought
update_thought() {
  local thought_id=$1
  echo "Updating thought with ID $thought_id..."
  curl -s -X PUT "$BASE_URL/thoughts/$thought_id" -H "Content-Type: application/json" -d '{
    "thought_description": "Updated thought description"
  }'
  echo -e "\n"
}

# Function to delete a thought
delete_thought() {
  local thought_id=$1
  echo "Deleting thought with ID $thought_id..."
  curl -s -X DELETE "$BASE_URL/thoughts/$thought_id"
  echo -e "\n"
}

# Function to create a memory
create_memory() {
  echo "Creating a new memory..."
  curl -s -X POST "$BASE_URL/memories/" -H "Content-Type: application/json" -d '{
    "user_id": 1,
    "memory_type": "happy",
    "memory_intensity": "high"
  }'
  echo -e "\n"
}

# Function to get all memories
get_memories() {
  echo "Getting all memories..."
  curl -s -X GET "$BASE_URL/memories/"
  echo -e "\n"
}

# Function to get a memory by ID
get_memory() {
  local memory_id=$1
  echo "Getting memory with ID $memory_id..."
  curl -s -X GET "$BASE_URL/memories/$memory_id"
  echo -e "\n"
}

# Function to update a memory
update_memory() {
  local memory_id=$1
  echo "Updating memory with ID $memory_id..."
  curl -s -X PUT "$BASE_URL/memories/$memory_id" -H "Content-Type: application/json" -d '{
    "memory_description": "Updated memory description"
  }'
  echo -e "\n"
}

# Function to delete a memory
delete_memory() {
  local memory_id=$1
  echo "Deleting memory with ID $memory_id..."
  curl -s -X DELETE "$BASE_URL/memories/$memory_id"
  echo -e "\n"
}

# Function to create a setting
create_setting() {
  echo "Creating a new setting..."
  curl -s -X POST "$BASE_URL/settings/" -H "Content-Type: application/json" -d '{
    "user_id": 1,
    "setting_type": "theme",
    "setting_value": "dark"
  }'
  echo -e "\n"
}

# Function to get all settings
get_settings() {
  echo "Getting all settings..."
  curl -s -X GET "$BASE_URL/settings/"
  echo -e "\n"
}

# Function to get a setting by ID
get_setting() {
  local setting_id=$1
  echo "Getting setting with ID $setting_id..."
  curl -s -X GET "$BASE_URL/settings/$setting_id"
  echo -e "\n"
}

# Function to update a setting
update_setting() {
  local setting_id=$1
  echo "Updating setting with ID $setting_id..."
  curl -s -X PUT "$BASE_URL/settings/$setting_id" -H "Content-Type: application/json" -d '{
    "setting_value": "light"
  }'
  echo -e "\n"
}

# Function to delete a setting
delete_setting() {
  local setting_id=$1
  echo "Deleting setting with ID $setting_id..."
  curl -s -X DELETE "$BASE_URL/settings/$setting_id"
  echo -e "\n"
}

# Function to create a trading card
create_trading_card() {
  echo "Creating a new trading card..."
  curl -s -X POST "$BASE_URL/trading_cards/" -H "Content-Type: application/json" -d '{
    "user_id": 1,
    "card_type": "meditation",
    "card_design": "serene",
    "realm_access_url": "http://example.com/realm",
    "qr_code_url": "http://example.com/qrcode"
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

# Run all tests
create_user
get_users
get_user 1
update_user 1

create_session
get_sessions
get_session 1
update_session 1

create_script
get_scripts
get_script 1
update_script 1

create_scene
get_scenes
get_scene 1
update_scene 1

create_element
get_elements
get_element 1
update_element 1

create_emotion
get_emotions
get_emotion 1
update_emotion 1

create_goal
get_goals
get_goal 1
update_goal 1

create_progress
get_all_progress
get_progress 1
update_progress 1

create_notification
get_notifications
get_notification 1
update_notification 1

create_thought
get_thoughts
get_thought 1
update_thought 1

create_memory
get_memories
get_memory 1
update_memory 1

create_setting
get_settings
get_setting 1
update_setting 1

create_trading_card
get_trading_cards
get_trading_card 1
update_trading_card 1

echo "API Testing Completed"
