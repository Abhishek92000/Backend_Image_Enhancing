# import cv2
# import torch
# from realesrgan import RealESRGANer
# from basicsr.archs.rrdbnet_arch import RRDBNet
#
# # Load the model
# model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
#                 num_block=23, num_grow_ch=32, scale=4)
#
# # Path to model weights
# model_path = 'RealESRGAN_x4plus.pth'
#
# # Initialize upsampler
# upsampler = RealESRGANer(
#     scale=4,
#     model_path=model_path,
#     model=model,
#     pre_pad=0,
#     half=torch.cuda.is_available()
# )
#
# # Read input image
# input_img = cv2.imread('04.jpg', cv2.IMREAD_COLOR)
#
# # Convert to RGB (Real-ESRGAN uses RGB)
# input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
#
# # Enhance (4x upscale)
# output_img_4x, _ = upsampler.enhance(input_img, outscale=1)
#
# # Convert back to BGR for saving
# output_img_4x = cv2.cvtColor(output_img_4x, cv2.COLOR_RGB2BGR)
#
# # Further upscale from 4x to 6x (1.5x more)
# height, width = output_img_4x.shape[:2]
# output_img_6x = cv2.resize(output_img_4x, (int(width * 1.5), int(height * 1.5)), interpolation=cv2.INTER_CUBIC)
#
# # Save final 6x image
# cv2.imwrite("output_6x.png", output_img_6x)


from flask import Flask, request, jsonify, send_from_directory
from feedback_server import send_feedback_email
from flask_cors import CORS
import os
# import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path='RealESRGAN_x4plus.pth',
    model=model,
    pre_pad=0,
    half=torch.cuda.is_available()
)


@app.route('/enhance', methods=['POST'])
def enhance():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image uploaded'}), 400

    file = request.files['image']
    filename = file.filename
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(upload_path)

    # Read image
    img = cv2.imread(upload_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        output_img, _ = upsampler.enhance(img, outscale=2)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

        output_filename = f"enhanced_{filename}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        cv2.imwrite(output_path, output_img)

        return jsonify({'success': True, 'filename': output_filename})
    except Exception as e:
        print(e)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)


@app.route('/send-feedback', methods=['POST'])
def send_feedback():
    data = request.json
    feedback = data.get('message', '')

    try:
        send_feedback_email(feedback)
        return jsonify({"success": True})
    except Exception as e:
        print(e)
        return jsonify({"success": False})


if __name__ == '__main__':
    app.run(debug=True)
