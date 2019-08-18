from flask import Flask,request, render_template, session
from flask_uploads import UploadSet, configure_uploads, IMAGES
import infer_on_single_image as code_base
import os
app = Flask(__name__)
app.secret_key = os.urandom(24)
photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'static/temp'
configure_uploads(app, photos)
upload_filename = ""

# Storing models to reduce load time
model_oxford = code_base.getModel()
model_paris = code_base.getModel(weights_file="./static/weights/paris_final.pth")
valid_img_oxford = code_base.getQueryNames(labels_dir="./static/data/oxbuild/gt_files/", 
                img_dir="./static/data/oxbuild/images/")
valid_img_paris = code_base.getQueryNames(labels_dir="./static/data/paris/gt_files/", 
                img_dir="./static/data/paris/images/")

# Main page
@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template("main.html", valid_img_oxford=valid_img_oxford, valid_img_paris=valid_img_paris)

# Validation page (from main page)
@app.route("/ImgSelected/<group_name>/<img_number>")
def evaluateValid(group_name,img_number):
    if (group_name=="paris"):
        filename = valid_img_paris[int(img_number)]
        similar_images, gt = code_base.inference_on_single_labelled_image_pca_web(model_paris,filename,labels_dir="./static/data/paris/gt_files/", 
                img_dir="./static/data/paris/images/",
                img_fts_dir="./static/fts_pca/paris/")
        prev_evaluated_images = similar_images
        session['prev_evaluated_images'] = prev_evaluated_images
        return render_template("img_selected.html", filename = filename, evaluated=prev_evaluated_images, gt=gt, valid=1)
    else:
        filename = valid_img_oxford[int(img_number)]
        similar_images, gt = code_base.inference_on_single_labelled_image_pca_web(model_oxford,filename)
        prev_evaluated_images = similar_images
        session['prev_evaluated_images'] = prev_evaluated_images
        return render_template("img_selected.html", filename = filename, evaluated=prev_evaluated_images, gt=gt, valid=1)

# Validation page repeated (directly from another validation page)
@app.route("/ImgSelected2/<img_number>")
def evaluateValid2(img_number):
    filename = session.get('prev_evaluated_images')[int(img_number)]
    similar_images= code_base.inference_on_single_labelled_image_pca_web_original(model_paris,filename)
    gt = [0]*60
    prev_evaluated_images = similar_images
    session['prev_evaluated_images'] = prev_evaluated_images
    return render_template("img_selected.html", filename = filename, evaluated=prev_evaluated_images, gt=gt, valid=0)

# Prediction page
@app.route("/ImgSelected/upload/", methods=['GET', 'POST'])
def evaluateNew():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        filename = '/static/temp/'+filename
        similar_images = code_base.inference_on_single_labelled_image_pca_web_original(model_paris,filename)
        gt = [0]*60
        prev_evaluated_images = similar_images
        session['prev_evaluated_images'] = prev_evaluated_images
        return render_template("img_selected.html", filename = filename, evaluated=prev_evaluated_images,gt=gt, valid=0)
    return render_template("img_selected.html", filename = "")

if __name__ == "__main__":
    app.run(debug=True)