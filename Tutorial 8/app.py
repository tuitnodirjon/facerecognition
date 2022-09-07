

from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import face_recognition
import numpy as np
app=Flask(__name__)

# Load a sample picture and learn how to recognize it.
# krish_image = face_recognition.load_image_file("Nodirjon/Nodirjon.png")
# krish_face_encoding = face_recognition.face_encodings(krish_image)[0]
#
# # Load a second sample picture and learn how to recognize it.
# bradley_image = face_recognition.load_image_file("Bradley/bradley.JPG")
# bradley_face_encoding = face_recognition.face_encodings(bradley_image)[0]
#
# # Create arrays of known face encodings and their names
# known_face_encodings = [
#     krish_face_encoding,
#     bradley_face_encoding
# ]
# known_face_names = [
#     "Nodirbek",
#     "Bradly"
# ]
import os
peoples = os.listdir('people')
known_face_encodings = []
known_face_names = []
for i in peoples:
    image = face_recognition.load_image_file("people/"+i)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(i.split('.')[0])

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
           
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
            

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/upload', methods=['POST', 'GET'])
def uploud():
    if request.method == 'POST':
        print(request.files, request.form)
        f = request.files['file']
        f.save(os.path.join('people', request.form['name']+'.'+f.filename.split('.')[-1]))
        return redirect(url_for('index'))
    return render_template('uploud.html')


if __name__=='__main__':
    app.run(debug=True)