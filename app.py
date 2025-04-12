from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from gradio import mount_gradio_app
from auth import init_csv, register_user, verify_user
from chat_interface import create_gradio_app  # ✅ Use the mountable version

# Initialize FastAPI and user system
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="supersecretkey")
init_csv()

# ✅ Mount Gradio app at `/gradio`
gradio_app = create_gradio_app()
app = mount_gradio_app(app, gradio_app, path="/gradio")

# --- Login/Registration Pages ---

@app.get("/", response_class=HTMLResponse)
def login_page():
    return """
    <h2>Login</h2>
    <form action="/login" method="post">
        Username: <input name="username"><br>
        Password: <input name="password" type="password"><br>
        <button type="submit">Login</button>
    </form>
    <p>New? <a href="/register">Register here</a></p>
    """

@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    if verify_user(username, password):
        return RedirectResponse("/gradio", status_code=302)
    return HTMLResponse("<h3>❌ Login failed. <a href='/'>Try again</a></h3>")

@app.get("/register", response_class=HTMLResponse)
def register_page():
    return """
    <h2>Create Account</h2>
    <form action="/register" method="post">
        Username: <input name="username"><br>
        Password: <input name="password" type="password"><br>
        <button type="submit">Register</button>
    </form>
    """

@app.post("/register")
def register(username: str = Form(...), password: str = Form(...)):
    if register_user(username, password):
        return HTMLResponse("<h3>✅ Account created. <a href='/'>Login now</a></h3>")
    return HTMLResponse("<h3>⚠️ Username exists. <a href='/register'>Try again</a></h3>")
