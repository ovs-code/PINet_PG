console.log('hello world')

const alertBox = document.getElementById('alert-box')
const imageBox = document.getElementById('image-box')
const imageForm = document.getElementById('image-form')
const confirmBtn = document.getElementById('confirm-btn')
const input = document.getElementById('id_file')

const csrf = document.getElementsByName('csrfmiddlewaretoken')

input.addEventListener('change', ()=>{
    alertBox.innerHTML = ""
    confirmBtn.classList.remove('not-visible')
    const img_data = input.files[0]
    const url = URL.createObjectURL(img_data)

    imageBox.innerHTML = `<img src="${url}" id="image" width="400px">`
    const image = document.getElementById('image')
    var $image = $('#image')
    console.log($image)

    const cropper = new Cropper(image, {
        autoCrop: true,
        autoCropArea: 0,
        initialAspectRatio: 176/256,
        cropBoxResizable: false,
        crop: function(event) {
            console.log(event.detail.x);
            console.log(event.detail.y);
            console.log(event.detail.width);
            console.log(event.detail.height);
            console.log(event.detail.rotate);
            console.log(event.detail.scaleX);
            console.log(event.detail.scaleY);
        },
        dragMode: 'move',
        movable: false
    });
    
    confirmBtn.addEventListener('click', ()=>{
        cropper.getCroppedCanvas({
            width: 176,
            height: 256,
            fillColor: 'white',
        }).toBlob((blob) => {
            console.log('confirmed')
            const fd = new FormData();
            // fd.append('csrfmiddlewaretoken', csrf[0].value)
            fd.append('image', blob, 'my-image.png');
            var inputPose = $("input[type='radio'][name='inputPose']:checked").val();
            fd.append('inputPose', inputPose);

            $.ajax({
                type:'POST',
                url: imageForm.action,
                enctype: 'multipart/form-data',
                data: fd,
                success: function(response){
                    console.log('success', response)
                    alertBox.innerHTML = `<div class="alert alert-success" role="alert">
                                            Successfully saved and cropped the selected image
                                        </div>`
                },
                error: function(error){
                    console.log('error', error)
                    alertBox.innerHTML = `<div class="alert alert-danger" role="alert">
                                            Ups...something went wrong
                                        </div>`
                },
                cache: false,
                contentType: false,
                processData: false,
            })
        })
    })
})

LIMB_SEQ = JSON.parse("[[5,7], [7, 9], [5, 6], [6, 8], [8, 10], [5, 11], [11, 12], [6, 12], [11, 13], [13, 15], [12, 14], [14, 16]]")

MISSING_VALUE = -1

function drawLine(ctx, from, to) {
    ctx.strokeStyle = 'white'
    ctx.beginPath();
    ctx.moveTo(from[1], from[0]);
    ctx.lineTo(to[1], to[0]);
    ctx.stroke();
}
// function for drawing the poses
function drawPose(canvas, pose){
    if(canvas.getContext){
        canvas.width = 176
        canvas.height = 256
        var ctx = canvas.getContext('2d');
        
        LIMB_SEQ.forEach(limb => {
            if(pose[limb[0]][0] === MISSING_VALUE | pose[limb[0]][1] === MISSING_VALUE | pose[limb[1]][0] === MISSING_VALUE | pose[limb[1]][1] === MISSING_VALUE) {
                return;
            } else {
                drawLine(ctx, pose[limb[1]], pose[limb[0]]);
            }
        })
        ctx.fillStyle = 'red';
        pose.forEach(limb => {
            if (limb[0] === MISSING_VALUE | limb[1] === MISSING_VALUE) {
                return;
            }
            ctx.fillRect(limb[1], limb[0], 5, 5);
        });
    }
}

const posesBox = document.getElementById('posesBox')
var numPoses = 1  // set this for the number of selectable poses

for (let index = 0; index < numPoses; index++) {
    var canvas = document.getElementById(`pose${index}`)
    var pose = JSON.parse(canvas.dataset.pose)

    drawPose(canvas, pose);
}