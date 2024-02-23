import gradio as gr
import requests

# Global variable to store the introductory message and the latest question
intro_message = "Ada yang bisa saya bantu?"
latest_question = ""

def chat_with_model(question, history, regenerate=False):
    global latest_question
    url = 'http://127.0.0.1:8000/chat'

    # Determine whether to use the new question or regenerate the response for the latest question
    actual_question = latest_question if regenerate else question
    if not regenerate:
        latest_question = question  # Update the latest question

    # If history starts with the intro message, remove it
    if history.startswith(intro_message):
        history = history[len(intro_message):].strip()

    response = requests.post(url, json={"question": actual_question})
    if response.status_code == 200:
        data = response.json()
        new_history = f"{history}\n\nUser: {actual_question}\nBot: {data['answer']}" if history else f"User: {actual_question}\nBot: {data['answer']}"
        return new_history, data['answer'], ""  # Return updated history, response, and clear question input
    else:
        new_history = f"{history}\n\nUser: {actual_question}\nBot: Error in getting response." if history else f"User: {actual_question}\nBot: Error in getting response."
        return new_history, "Error: Unable to get response from the model.", ""  # Return updated history, error response, and clear question input

def handle_feedback(feedback):
    # Process feedback here
    if feedback == "Good response":
        response_message = "Terima kasih untuk saran nya!"
    else:
        response_message = "Kita minta maaf dan berusaha untuk mengembangkan jawaban yang lebih baik."

    return response_message

# Define a function to return the profile content
def get_profile_content():
    # HTML content for the profile page
    profile_html = """
    <div style='text-align: center; padding: 20px;'>
        <h2>Audi Chandra</h2>
        <p><strong>About Me:</strong><br>
        I am Audi Chandra, a passionate AI enthusiast and software developer with a keen interest in building intelligent systems. My work revolves around designing and implementing AI models to solve real-world problems.</p>
        <p><strong>Contact Information:</strong><br>
        Email: audichandra94@gmail.com<br>
        LinkedIn: <a href='https://www.linkedin.com/in/audi-chandra-131a9864/' target='_blank'>www.linkedin.com/in/audi-chandra-131a9864/</a><br>
        GitHub: <a href='https://github.com/audichandra' target='_blank'>github.com/audichandra</a></p>
    </div>
    """
    return profile_html

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Chat"):
            with gr.Column():
                history_output = gr.TextArea(label="Chat History", value=intro_message, interactive=False, lines=6)
                question_input = gr.Textbox(label="Question", placeholder="Silahkan ketikkan pertanyaan anda disini", lines=2)
                send_button = gr.Button("Send")
                response_output = gr.Textbox(label="Latest Response")
                feedback = gr.Radio(["Good response", "Bad response"], label="Is it a good response?")
                feedback_message = gr.Textbox(label="Feedback Message", visible=True, interactive=False)
                submit_feedback = gr.Button("Submit Feedback")

            send_button.click(
                chat_with_model, 
                inputs=[question_input, history_output],
                outputs=[history_output, response_output, question_input]
            )

            
            submit_feedback.click(
                handle_feedback,
                inputs=feedback,
                outputs=feedback_message
            ) 
        
        # Profile tab
        with gr.TabItem("Profile"):
            gr.HTML(value=get_profile_content())
    
    #regenerate_response = gr.Button("Regenerate Response")
    #regenerate_response.click(
        #lambda _, hist: chat_with_model("", hist, regenerate=True),
        #inputs=[question_input, history_output],
        #outputs=[history_output, response_output, question_input])
    
    gr.Markdown("<div class='copyright'>Copyright &copy; Audi Chandra</div>")

            
demo.launch(share=True)