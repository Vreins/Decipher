from django.urls import path
from .views import index, upload_files, chat_message, retrieve_chat_history, login_view, register, create_new_session, logout_user, soft_delete_chat, soft_select_chat

urlpatterns = [
    path('', index, name='index'),
    path('upload_files', upload_files, name='upload_files'),
    path('chat_message', chat_message, name='chat_message'),
    path('retrieve_chat_history', retrieve_chat_history, name='retrieve_chat_history'),
    path('login/', login_view, name='login'),
    path('register/', register, name='register'),  # Add this line
    path('create_new_session', create_new_session, name='create_new_session'),  # Add this line
    path('logout_user', logout_user, name='logout_user'),
    path('soft_delete_chat', soft_delete_chat, name='soft_delete_chat'),
    path('soft_select_chat', soft_select_chat, name='soft_select_chat')
]