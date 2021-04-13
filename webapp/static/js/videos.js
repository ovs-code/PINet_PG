const alertBox = document.getElementById('alert-box')
const imageBox = document.getElementById('image-box')
const imageForm = document.getElementById('image-form')
const confirmBtn = document.getElementById('confirm-btn')
const input = document.getElementById('id_file')
const cropBox = document.getElementById('crop-box')

const csrf = document.getElementsByName('csrfmiddlewaretoken')
var url;

input.addEventListener('change', ()=>{
    function cropIt() {
        console.log('crop it');
        cropper.getCroppedCanvas({
            width: 176,
            height: 256,
            fillColor: 'white',
        }).toBlob((blob) => {
            url = URL.createObjectURL(blob)
            // instead display the cropped image
            cropBox.innerHTML = `<img class="mx-auto d-block" src="${url}" id="crop">`
        })
        // show the cropBox
        cropBox.style.display = 'block';

        // hidde the cropper and the form
        imageBox.style.display = 'none';
        imageForm.style.display = 'none';

        confirmBtn.innerText = 'Recrop/Upload Image';
        confirmBtn.removeEventListener('click', cropIt);
        confirmBtn.addEventListener('click', recropIt);
        $('html,body').animate({
            scrollTop: $('#pose-heading').offset().top
        }, 'slow')
        document.getElementById('submit-btn').disabled = false;
    };

    function recropIt() {
        // hide the currently cropped image
        cropBox.style.display = 'none';

        // show the cropper and the form
        imageBox.style.display = 'block';
        imageForm.style.display = 'block';

        confirmBtn.innerText = 'Crop Image';
        confirmBtn.removeEventListener('click', recropIt);
        confirmBtn.addEventListener('click', cropIt);
        $('html,body').animate({
            scrollTop: $('#upload-heading').offset().top
        }, 'slow')
    };
    alertBox.innerHTML = ""
    confirmBtn.classList.remove('not-visible')
    confirmBtn.style = null;
    const img_data = input.files[0]
    url = URL.createObjectURL(img_data)

    imageBox.innerHTML = `<img src="${url}" id="image" width="400px">`
    const image = document.getElementById('image')

    const cropper = new Cropper(image, {
        autoCrop: true,
        autoCropArea: 0,
        initialAspectRatio: 176 / 256,
        cropBoxResizable: false,
        crop: function (event) {
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

    $('html,body').animate({
        scrollTop: $('#upload-heading').offset().top
    }, 1000)

    confirmBtn.addEventListener('click', cropIt);
})

// make the first radiobutton checked
$('input:radio[name=inputPose]:first').attr('checked', true);

const posesBox = document.getElementById('posesBox')

//Display the result image
const submitBtn = document.getElementById('submit-btn')
submitBtn.addEventListener('click', async () => {
    let blob = await fetch(url).then(r => r.blob());
    const fd = new FormData();
    // fd.append('csrfmiddlewaretoken', csrf[0].value)
    fd.append('image', blob, 'my-image.png');
    var inputPose = $("input[type='radio'][name='inputPose']:checked").val();
    fd.append('inputPose', inputPose);
    let id;

    $.ajax({
        type: 'POST',
        url: imageForm.action,
        enctype: 'multipart/form-data',
        data: fd,
        error: function (error) {
            alertBox.innerHTML = `<div class="alert alert-danger" role="alert">
                                            Ups...something went wrong
                                        </div>`
        },
        cache: false,
        contentType: false,
        processData: false,
    }).done(
        function(data) {
            console.log(data)
            id = data;
        },
        submitBtn.innerHTML = `<span class="spinner-border spinner-border-sm"></span>Transforming...`,
        submitBtn.disabled = true,
        console.log('sucessfully sent to backend'),
        // create the spinning wheel
        setTimeout(async () => {
            console.log(id)
            const progressChecker = async () => {
                // check every second til complete
                status = await fetch('/api/status/' + id).then(r => r.text());
                if (status !== '[completed]') {
                    if (status === '[failed]') {
                        waitMessage.innerText = 'Backend said this failed. Sorry :('
                    } else {
                        setTimeout(progressChecker, 1000);
                    }
                } else {
                    let result = await fetch('/api/result/' + id).then(r => r.json());
                    // display the response image
                    const output = document.getElementById('output-container');
                    output.style.display = 'block';
                    const targetContainer = document.getElementById('target-image');
                    targetContainer.innerHTML = `<video width="100%" height="100%" autoplay loop>
						<source src=${"/static/videos/generated/" + result['target_image']} type="video/mp4">
							Your browser does not support the video tag.
					</video>`
                    submitBtn.innerHTML = `Transform`;
                    submitBtn.disabled = false;
                    $('html,body').animate({
                        scrollTop: $('#output-container').offset().top
                    }, 1000)
                }
            };
            setTimeout(progressChecker, 1000);
        }, 0)
    )
})