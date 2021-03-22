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
        minCropBoxWidth: 176,
        minCropBoxHeight: 256,
        cropBoxResizable: false,
        crop: function(event) {
            console.log(event.detail.x);
            console.log(event.detail.y);
            console.log(event.detail.width);
            console.log(event.detail.height);
            console.log(event.detail.rotate);
            console.log(event.detail.scaleX);
            console.log(event.detail.scaleY);
        }
    });
    
    confirmBtn.addEventListener('click', ()=>{
        cropper.getCroppedCanvas({
            width: 176,
            height: 256
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