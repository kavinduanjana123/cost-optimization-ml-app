from flask import Blueprint, render_template, request, redirect, url_for, session

auth_bp = Blueprint('auth', __name__)

users = {"admin": "1234"}

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['user'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@auth_bp.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('auth.login'))
