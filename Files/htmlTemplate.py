css = '''
<style>
.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    display: flex;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    transition: 0.3s;
}

.chat-message.user {
    background-color: #4a69bd;
    color: white;
    align-items: center;
}

.chat-message.bot {
    background-color: #78e08f;
    color: black;
    align-items: center;
}

.chat-message .avatar {
    width: 15%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.chat-message .avatar img {
    max-width: 50px;
    max-height: 50px;
    border-radius: 50%;
    object-fit: cover;
}

.chat-message .message {
    width: 85%;
    padding: 0 1rem;
    font-size: 1rem;
    font-family: 'Arial', sans-serif;
}

.chat-message:hover {
    box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
}

/* Styling for the expander */
.st-ba {
    border-radius: 10px;
    border: 1px solid #ccc;
    margin-bottom: 1rem;
}

.st-bb {
    background-color: #f0f0f0;
    padding: 0.5rem 1rem;
}

.st-be {
    color: #333;
    font-weight: bold;
    cursor: pointer;
}

.st-be:hover {
    color: #555;
}

.st-bo {
    margin-top: 0.5rem;
}

.st-bd {
    padding: 0.5rem 1rem;
}
</style>
'''

bot_template = '''
<div class="chat-message bot"">
    <div class="avatar">
        <img src="https://www.vhv.rs/dpng/d/552-5522826_thumb-image-answer-icon-png-transparent-png.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user" style="text-align: left;">
    <div class="avatar">
        <img src="https://www.vhv.rs/dpng/d/412-4120955_t-55cb074191589-question-image-55cb074191496-2-q-train.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
