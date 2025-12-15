from flask import Flask, request, jsonify
import smtplib
from email.mime.text import MIMEText
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def send_feedback_email(feedback):
    sender_email = os.getenv("SENDER_EMAIL")
    receiver_email = os.getenv("SENDER_EMAIL")  # Send to yourself
    password = os.getenv("EMAIL_PASSWORD")

    msg = MIMEText(feedback)
    msg['Subject'] = 'New Feedback'
    msg['From'] = sender_email
    msg['To'] = receiver_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
