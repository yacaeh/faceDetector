// elements
const s = document.getElementById('objDetect');
const rInfo = document.getElementById('captureInfo');
const rAlert = document.getElementById('recAlert');

// attribs
const sourceVideo = s.getAttribute("data-source");  //the source video to use
const uploadWidth = s.getAttribute("data-uploadWidth") || 640; //the width of the upload file
const mirror = s.getAttribute("data-mirror") || false; //mirror the boundary boxes
const scoreThreshold = s.getAttribute("data-scoreThreshold") || 0.5;
const detectUrl = window.location.origin + '/detect';
const updateUrl = window.location.origin + '/train';

//Video element selector
v = document.getElementById(sourceVideo);

//for starting events
let isPlaying = false,
gotMetadata = false;
let isCaptureExample = false;
let examplesNum = 0;
let maxExamples = 20;
let exampleSize = 160;

//Canvas setup

//create a canvas to grab an image for upload
let imageCanvas = document.createElement('canvas');
let imageCtx = imageCanvas.getContext("2d");

//let exCanvas = document.createElement('canvas');
let exCanvas = document.getElementById("exCanvas")
let exCtx = exCanvas.getContext("2d");
//document.getElementById('examplesPic').appendChild(exCtx);

//create a canvas for drawing object boundaries
let drawCanvas = document.createElement('canvas');

var rect = v.getBoundingClientRect();
console.log("Video box:", rect.top, rect.right, rect.bottom, rect.left);

document.getElementById('videoDiv').appendChild(drawCanvas);
//document.getElementById('buttons').style.top = rect.bottom + 10;
let drawCtx = drawCanvas.getContext("2d");

function uploadScale() {
    return v.videoWidth > 0 ? uploadWidth / v.videoWidth : 0;
}

let previousName ='';
let faceCount = 0;

//draw boxes and labels on each detected object
function drawBoxes(objects) {

    //clear the previous drawings
    drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);

    //filter out objects that contain a class_name and then draw boxes and labels on each
    objects.forEach(face => {
        let scale = uploadScale();
        let _x = face.x / scale;
        let y = face.y / scale;
        let width = face.w / scale;
        let height = face.h / scale;
        //flip the x axis if local video is mirrored
        if (mirror) {
            x = drawCanvas.width - (_x + width)
        } else {
            x = _x
        }

        let rand_conf = face.confidence.toFixed(2);
        let title = "" + rand_conf + "";
        if (face.name != "unknown") {
            drawCtx.strokeStyle = "magenta";
            drawCtx.fillStyle = "magenta";
            title += ' - ' + face.name
            if (face.predict_proba > 0.0 ) {
                title += "[" + face.predict_proba.toFixed(2) + "]";
            }
            if(face.name != previousName){
                previousName = face.name;
                faceCount =1;
            }
            else if(face.name == previousName){
                faceCount++;
                if(faceCount>2){
                    drawCtx.strokeStyle = "green";
                    drawCtx.fillStyle = "green";
                    title += ' - ' + face.name +'로그인!'
                            logIn(face.name);
                    faceCount = 0;
                    previousName='';
                }
            }
            console.log("faceCount:"+faceCount);
        } else {
            drawCtx.strokeStyle = "cyan";
            drawCtx.fillStyle = "cyan";
            title += '얼굴을 좀더 똑바로 비춰주세요';
        }
        drawCtx.fillText(title , x + 5, y - 5);
        drawCtx.strokeRect(x, y, width, height);

        if(isCaptureExample && examplesNum < maxExamples) {
            console.log("capure example: ", examplesNum)
            //Some styles for the drawcanvas
            exCtx.drawImage(imageCanvas,
                face.x, face.y, face.w, face.h,
                (examplesNum%10) * exampleSize, Math.floor(examplesNum/10)*exampleSize,
                exampleSize, exampleSize);
                console.log("Math.floor(examplesNum/10):"+Math.floor(examplesNum/10));

            examplesNum += 1;

            if(examplesNum == maxExamples) {
                stopCaptureExamples();
            }
        }

    });
}

//Add file blob to a form and post
function postFile(file) {

    //Set options as form data
    let formdata = new FormData();
    formdata.append("image", file);
    formdata.append("threshold", scoreThreshold);

    let xhr = new XMLHttpRequest();
    xhr.open('POST', detectUrl, true);
    xhr.onload = function () {
        if (this.status === 200) {
            let objects = JSON.parse(this.response);

            //draw the boxes
            drawBoxes(objects);

            //Save and send the next image
            imageCtx.drawImage(v, 0, 0, v.videoWidth, v.videoHeight, 0, 0, uploadWidth, uploadWidth * (v.videoHeight / v.videoWidth));
            imageCanvas.toBlob(postFile, 'image/jpeg');
        }
        else {
            console.error(xhr);
        }
    };
    xhr.send(formdata);
}

function postExamplesFile(file) {
    //Set options as form data
    let formdata = new FormData();
    formdata.append("image", file);
    formdata.append("num", examplesNum);
    formdata.append("size", exampleSize);
    name = document.getElementById('inputName').value;
    specialText = document.getElementById('specialText').value;
    specialText2 = document.getElementById('specialText2').value;
    formdata.append("name", name);
    let personObj = {name:name,specialText:specialText,specialText2:specialText2}
    register(personObj);
    let xhr = new XMLHttpRequest();
    xhr.open('POST', updateUrl, true);
    xhr.onload = function () {
        if (this.status === 200) {
            let objects = JSON.parse(this.response);
            console.log(objects);
            alert(
            'Model updated with person: ' + name + ' \n' +
            'Now model have examples for: \n\n' +
            objects.map( i => '' + i.name + ' - train examples: ' + i.train_examples ).join('\n\n')
            )
        }
        else {
            console.error(xhr);
        }
    };
    xhr.send(formdata);
}

//Start object detection
function startObjectDetection() {

    console.log("starting object detection");

    //Set canvas sizes base don input video
    drawCanvas.width = v.videoWidth;
    drawCanvas.height = v.videoHeight;

    imageCanvas.width = uploadWidth;
    imageCanvas.height = uploadWidth * (v.videoHeight / v.videoWidth);

    //Some styles for the drawcanvas
    drawCtx.lineWidth = 4;
    drawCtx.strokeStyle = "cyan";
    drawCtx.font = "20px Verdana";
    drawCtx.fillStyle = "cyan";

    //Save and send the first image
    imageCtx.drawImage(v, 0, 0, v.videoWidth, v.videoHeight, 0, 0, uploadWidth, uploadWidth * (v.videoHeight / v.videoWidth));
    imageCanvas.toBlob(postFile, 'image/jpeg');

}

//Capture examples for training
function captureExamples() {
    console.log("staring capturing sprites...")
    rInfo.hidden = false;
    rAlert.hidden = false;
    isCaptureExample = true;
    examplesNum = 0;
    exCtx.clearRect(0, 0, exCanvas.width, exCanvas.height);
    document.getElementById('updateModel').hidden = true;

}

function stopCaptureExamples() {
    rAlert.hidden = true;
    document.getElementById('updateModel').hidden = false;
}

function updateModel() {
    console.log("updating model...")
    //Save and send the next image
    exCanvas.toBlob(postExamplesFile, 'image/jpeg');
}

// EVENTS

//check if metadata is ready - we need the video size
v.onloadedmetadata = () => {
    console.log("video metadata ready");
    gotMetadata = true;
    if (isPlaying)
        startObjectDetection();
};

//see if the video has started playing
v.onplaying = () => {
    console.log("video playing");
    isPlaying = true;
    if (gotMetadata) {
        startObjectDetection();
    }
};


window.onload = () => {
    document.getElementById("buttonCapture").onclick = () => {
        captureExamples();
       };
    document.getElementById("updateModel").onclick = () => {
        let n = document.getElementById('inputName')
        if (n.value) {
            updateModel();
         } else {
            alert("Please provide name!");
         }
        return false;
    };

};

function logIn(name) {
    console.log("login!"+name);
    let url = 'https://192.168.124.100:5555/login';
    let xhr = new XMLHttpRequest();
    var data = {name:name};
    xhr.open('POST', url, true);
    xhr.setRequestHeader('Content-Type', 'application/json; charset=UTF-8');
    xhr.onload = function () {
    if (this.status === 200) {
        let objects = this.response;
        console.log(objects);
        var audio = document.getElementById('audio');
        var mp3File = objects;//'https://cache-a.oddcast.com/c_fs/b58315278378d0069fd6a14178355ed3.mp3?engine=3&language=13&voice=9&text=%EC%95%88%EB%85%95%ED%95%98%EC%84%B8%EC%9A%94%20%EC%A2%80%EB%B9%84%EB%8B%98%3F%20%ED%99%98%EC%98%81%ED%95%A9%EB%8B%88%EB%8B%A4.%20%EB%84%88%20%EB%98%90%20%EB%8A%A6%EA%B2%8C%20%EC%99%94%EB%8B%A4.%20%EC%88%A8%EC%A7%80%EA%B3%A0%20%EC%8B%B6%EB%83%90%3F&useUTF8=1'
        audio.src = mp3File;
        audio.play();    
    }
    else {
        console.log("errro!");
        console.error(xhr);
        var audio = document.getElementById('audio');
        var mp3File = 'https://cache-a.oddcast.com/c_fs/45dea4ee2e656536a6cb7aebde585260.mp3?engine=3&language=13&voice=7&text=%EB%B6%88%EC%8C%8D%ED%95%9C%20%EC%9A%B0%EB%A6%AC%20%EC%A2%80%EB%B9%84%2C%20%EB%98%90%20%EB%B2%84%EA%B7%B8%20%ED%84%B0%EC%A1%8C%EB%84%A4%3F%20%EC%98%A4%EB%8A%98%EB%8F%84%20%EC%A7%91%EC%97%90%20%EA%B0%80%EA%B8%B4%20%EA%B8%80%EB%A0%80%EB%8B%A4.&useUTF8=1'
        audio.src = mp3File;
        audio.play();    
    }
};
xhr.send(JSON.stringify(data));
}

function register(obj) {
    console.log("register!"+obj);
    let url = 'https://192.168.124.100:5555/register';
    let xhr = new XMLHttpRequest();
    var data = obj;
    console.log(data);
    xhr.open('POST', url, true);
    xhr.setRequestHeader('Content-Type', 'application/json; charset=UTF-8');
    xhr.onload = function () {
    if (this.status === 200) {
        let objects = this.response;
        console.log(objects);
        var audio = document.getElementById('audio');
        var mp3File = objects;//'https://cache-a.oddcast.com/c_fs/b58315278378d0069fd6a14178355ed3.mp3?engine=3&language=13&voice=9&text=%EC%95%88%EB%85%95%ED%95%98%EC%84%B8%EC%9A%94%20%EC%A2%80%EB%B9%84%EB%8B%98%3F%20%ED%99%98%EC%98%81%ED%95%A9%EB%8B%88%EB%8B%A4.%20%EB%84%88%20%EB%98%90%20%EB%8A%A6%EA%B2%8C%20%EC%99%94%EB%8B%A4.%20%EC%88%A8%EC%A7%80%EA%B3%A0%20%EC%8B%B6%EB%83%90%3F&useUTF8=1'
        audio.src = mp3File;
        audio.play();    
    }
    else {
        console.log("errro!");
        console.error(xhr);
        var audio = document.getElementById('audio');
        var mp3File = 'https://cache-a.oddcast.com/c_fs/45dea4ee2e656536a6cb7aebde585260.mp3?engine=3&language=13&voice=7&text=%EB%B6%88%EC%8C%8D%ED%95%9C%20%EC%9A%B0%EB%A6%AC%20%EC%A2%80%EB%B9%84%2C%20%EB%98%90%20%EB%B2%84%EA%B7%B8%20%ED%84%B0%EC%A1%8C%EB%84%A4%3F%20%EC%98%A4%EB%8A%98%EB%8F%84%20%EC%A7%91%EC%97%90%20%EA%B0%80%EA%B8%B4%20%EA%B8%80%EB%A0%80%EB%8B%A4.&useUTF8=1'
        audio.src = mp3File;
        audio.play();    
    }
};
xhr.send(JSON.stringify(data));
}
