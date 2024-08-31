if (navigator.serviceWorker) {
    navigator.serviceWorker.register("../sw.js")
        .then((reg) => {
            console.log("file registerd", reg)

        })
        .catch((err) => {
            console.log("error", err)
        })
}
const topOverlay = document.querySelector('.top-overlay');
const bottomOverlay = document.querySelector('.bottom-overlay');
const windowHeight = window.innerHeight;

window.addEventListener('scroll', () => {
    const scrollPosition = window.scrollY;
    const topOpacity = Math.min(scrollPosition / (windowHeight / 2), 0.8);
    const bottomOpacity = Math.min((document.body.clientHeight - scrollPosition - windowHeight) / (windowHeight / 2), 0.8);
    topOverlay.style.opacity = topOpacity;
    bottomOverlay.style.opacity = bottomOpacity;
});


const imageContainer = document.querySelector('.image-container');
const images = Array.from(imageContainer.children);
function refreshDeleteIcons() {
    setInterval(function () {
        deleteIcons = document.querySelectorAll('.delete-icon');
    }, 1000);
}
function numberOfCol() {
    box = document.querySelector('.image-container');
    directChildren = box.children.length;
    return directChildren;
}
var deleteIcons = document.querySelectorAll('.delete-icon');
const undoButton = document.querySelector('.undo-button');
let deletedImages = [];

// Add click event listener to delete icons
imageContainer.addEventListener('click', (e) => {
    if (e.target.classList.contains('delete-icon')) {
        const image = e.target.parentNode;
        const imageIndex = images.indexOf(image);
        const imageHTML = image.outerHTML;
        deletedImages.unshift({ index: imageIndex, html: imageHTML });
        imageContainer.removeChild(image);
        // updateImagePositions();
    }
});

// Add click event listener to undo button
undoButton.addEventListener('click', () => {
    const restoredImage = deletedImages.shift();
    if (restoredImage) {
        imageContainer.insertAdjacentHTML('beforeend', restoredImage.html);
        const restoredImageElement = imageContainer.lastChild;
        restoredImageElement.querySelector('.delete-icon').addEventListener('click', (e) => {
            const image = e.target.parentNode;
            const imageIndex = images.indexOf(image);
            const imageHTML = image.outerHTML;
            deletedImages.unshift({ index: imageIndex, html: imageHTML });
            imageContainer.removeChild(image);
            updateImagePositions();
            refreshDeleteIcons();
        });
        imageContainer.insertBefore(restoredImageElement, imageContainer.children[restoredImage.index]);
        updateImagePositions();
        refreshDeleteIcons();
    }
    // updateImagePositions();
    setTimeout(() => {
        if (restoredImageElement) {
            restoredImageElement.querySelector('.delete-icon').removeEventListener('click');
        }
    }, 100);
});

// Function to update image positions
function updateImagePositions() {
    images.forEach((image, index) => {
        const position = index + 1;
        image.querySelector('input').value = `${position}`;
    });
}

// Drag and drop functionality
let draggedImage = null;

imageContainer.addEventListener('dragstart', (e) => {
    const image = e.target.closest('.image');
    if (image) {
        draggedImage = image;
        image.classList.add('dragging');
        // updateImagePositions()
    }
});

imageContainer.addEventListener('dragend', (e) => {
    if (draggedImage) {
        draggedImage.classList.remove('dragging');
        draggedImage = null;
        // updateImagePositions()
    }
});

imageContainer.addEventListener('dragover', (e) => {
    e.preventDefault();
    const image = e.target.closest('.image');
    if (draggedImage && image && draggedImage !== image) {
        const draggedIndex = images.indexOf(draggedImage);
        const dropIndex = images.indexOf(image);
        if (draggedIndex < dropIndex) {
            imageContainer.insertBefore(draggedImage, image.nextSibling);
        } else {
            imageContainer.insertBefore(draggedImage, image);
        }
        images.splice(draggedIndex, 1);
        images.splice(dropIndex, 0, draggedImage);
        // updateImagePositions()
    }
});

imageContainer.addEventListener('drop', (e) => {
    e.preventDefault();
    // updateImagePositions()
});

//Camera Functionality
const video = document.getElementById('video');
const captureButton = document.getElementById('capture');
const startCameraButton = document.getElementById('start-camera')
const stopCameraButton = document.getElementById('stop-camera');
let isCameraActive = false;


// Add this EventListener to chech if the browser support camera then start it
// لما بدوس على زرار بدا الكاميرا الفيديو بيشتغل و يظهر شوية زراير كانت مختفية
startCameraButton.addEventListener('click', () => {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
            video.srcObject = stream;
            video.play();
            video.style.display = "block";
            stopCameraButton.style.display = "block";
            captureButton.style.display = "block";
            startCameraButton.style.display = "none";
        });
});


// Add this EventListener to call the stopCamera function
// لما بدوس على زرار قفل الكاميرا  بينادي على فانكشن الفيديو بيتقفل و يخفي الزراير الخاصة بيه
stopCameraButton.addEventListener('click', () => {
    stopCamera();
});


// Add this function to show hidden buttons 
// فانكشن بتظهر الزراير المختفية
function showCropButton() {
    const cropDescription = document.getElementById('crop-description');
    const saveButton = document.getElementById('crop-button');
    const clear = document.getElementById('clear');
    saveButton.style.display = "block";
    cropDescription.style.display = "block";
    clear.style.display = "block";
}
const cropDescription2 = document.getElementById('crop-description2');
const saveButton2 = document.getElementById('crop-button2');
// const clear2 = document.getElementById('clear2');
function showCropButton2() {
    const cropDescription2 = document.getElementById('crop-description2');
    const saveButton2 = document.getElementById('crop-button2');
    // const clear2 = document.getElementById('clear2');
    saveButton2.style.display = "block";
    cropDescription2.style.display = "block";
    // clear2.style.display = "block";
}

// Add this function to stop the camera and remove the reserved space of the video element
// فانكشن بتخفي الكاميرا 
function stopCamera() {
    video.srcObject.getTracks()[0].stop();
    video.srcObject = null;
    video.style.display = "none";
    stopCameraButton.style.display = "none";
    captureButton.style.display = "none";
    startCameraButton.style.display = "block";
    isCameraActive = false;
}


// Add this EventListener to capture image from the video and enable all hidden button and crop
// لما بدوس على زرار الالتقاط بيكرت صورة من الفيديو و بيمسح اي صورة كانت موجودة قديمة و بيشغل الكروب و هكذا
captureButton.addEventListener('click', () => {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL('image/png');
    const image = new Image();
    const image2 = new Image();
    image.src = dataURL;
    image2.src = dataURL;
    const imageDiv = document.createElement('div');
    const imageDiv2 = document.createElement('div');
    imageDiv.classList.add('image-preview-item');
    imageDiv2.classList.add('image-preview-item2');
    fileInput.value = "";
    imageDiv.appendChild(image);
    imageDiv2.appendChild(image2);
    // remove existing images
    // while (imagePreview.firstChild) {
    //     imagePreview.removeChild(imagePreview.firstChild);
    // }
    imagePreview.innerHTML = "";
    imagePreview.appendChild(imageDiv);
    imagePreview2.innerHTML = "";
    imagePreview2.appendChild(imageDiv2);
    showCropButton();
    showCropButton2();
    let croppedImageData = null;
    let croppedImageData2 = null;
    // Add click event listener to start cropping process
    imageDiv.addEventListener('click', () => {
        const cropper = new Cropper(image, {
            // set the aspect ratio of the crop box
            crop: (event) => {
                // do something with the crop data (e.g. save it to a hidden input field)
                console.log(event.detail.x, event.detail.y, event.detail.width, event.detail.height);
                croppedImageData = cropper.getCroppedCanvas().toDataURL();
            },
            ready: () => {
                // show the crop box once the cropper is ready
                cropper.setCropBoxData({ width: image.naturalWidth, height: image.naturalHeight });
            }
        });
    });
    imageDiv2.addEventListener('click', () => {
        const cropper2 = new Cropper(image2, {
            // set the aspect ratio of the crop box
            crop: (event) => {
                // do something with the crop data (e.g. save it to a hidden input field)
                console.log(event.detail.x, event.detail.y, event.detail.width, event.detail.height);
                croppedImageData2 = cropper2.getCroppedCanvas().toDataURL();
            },
            ready: () => {
                // show the crop box once the cropper is ready
                cropper2.setCropBoxData({ width: image2.naturalWidth, height: image2.naturalHeight });
            }
        });
    });
    const cropButton = document.getElementById('crop-button');
    const errorCROP = document.getElementById("crop-error");
    const cropContainer = document.getElementById('cropped-image');
    const cropContainer2 = document.getElementById('cropped-image2');
    // const splitButton = document.getElementById('split-button');
    cropButton.addEventListener('click', () => {
        if (croppedImageData) {
            // errorCROP.style.display = "none"
            // splitButton.style.display = 'block';
            const croppedImage = new Image();
            croppedImage.src = croppedImageData;
            const croppedImageDiv = document.createElement('div');
            croppedImageDiv.appendChild(croppedImage);
            cropContainer.innerHTML = "";
            cropContainer.appendChild(croppedImageDiv);
            // const croppedImage2 = new Image();
            // croppedImage2.src = croppedImageData;
            // const croppedImageDiv2 = document.createElement('div');
            // croppedImageDiv2.appendChild(croppedImage2);
            // cropContainer2.innerHTML = "";
            // cropContainer2.appendChild(croppedImageDiv2);

        } else {
            errorCROP.innerText = 'no image selected to crop';
            errorCROP.style.display = 'block';
            setTimeout(function () {
                errorCROP.style.opacity = 1;
            }, 100); //0.1 second

            setTimeout(function () {
                errorCROP.style.opacity = 0;
            }, 2000); //2 second

            setTimeout(function () {
                errorCROP.style.display = 'none';
            }, 3000); //3 second

        }
    });
    const cropButton2 = document.getElementById('crop-button2');
    const errorCROP2 = document.getElementById("crop-error2");

    cropButton2.addEventListener('click', () => {
        if (croppedImageData2) {
            errorCROP2.style.display = "none"
            directChildren = numberOfCol();
            const croppedImage = new Image();
            croppedImage.src = croppedImageData2;
            const inputDiv = document.createElement('input');
            inputDiv.type = "number";
            inputDiv.value = directChildren + 1;
            inputDiv.classList.add('image-number-input');

            const img = document.createElement('img');
            img.src = croppedImageData2;

            const deleteDiv = document.createElement('div');
            deleteDiv.classList.add('delete-icon');

            const croppedImageDiv = document.createElement('div');


            croppedImageDiv.classList.add('image');
            croppedImageDiv.appendChild(inputDiv);
            croppedImageDiv.appendChild(img);
            croppedImageDiv.appendChild(deleteDiv);
            // document.getElementById('cropped-image').innerHTML = "";
            document.querySelector('.image-container').appendChild(croppedImageDiv);

            // const croppedImage2 = new Image();
            // croppedImage2.src = croppedImageData;
            // const croppedImageDiv2 = document.createElement('div');
            // croppedImageDiv2.appendChild(croppedImage2);
            // document.getElementById('cropped-image2').innerHTML = "";
            // document.getElementById('cropped-image2').appendChild(croppedImageDiv2);
        } else {
            errorCROP2.innerText = 'no image selected to crop';
            errorCROP2.style.display = 'block';
            setTimeout(function () {
                errorCROP2.style.opacity = 1;
            }, 100);

            setTimeout(function () {
                errorCROP2.style.opacity = 0;
            }, 2000);

            setTimeout(function () {
                errorCROP2.style.display = 'none';
            }, 3000);

        }
    });
});
// نهاية فانكشن الالتقاط



//drag & drop functionality
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const imagePreview = document.getElementById('image-preview');
const imagePreview2 = document.getElementById('image-preview2');

// Add this EventListener to drag images to zone
// بيحس لما بيعمل دراج صورة فوق المكان
dropZone.addEventListener('dragover', (event) => {
    event.preventDefault();
    dropZone.classList.add('dragover');
});


// Add this EventListener to remove the class style when dragleave
// بيحس لما بشيل الصورة من فوق المكان
dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});


// Add this EventListener to remove the class style when dropped and send the image to function to handle it like the camera 
// بيحس لما بعمل دروب 
dropZone.addEventListener('drop', (event) => {
    fileInput.value = "";
    event.preventDefault();
    const files = event.dataTransfer.files;
    handleFiles(files);
    dropZone.classList.remove('dragover');
});


// Add this EventListener to change input image and send the image to function to handle it like the camera 
// بيحس لو غيرت الصورة 
fileInput.addEventListener('change', () => {
    const files = fileInput.files;
    handleFiles(files);
});


const cropButton = document.getElementById('crop-button');


// Add this function to handle image from the input or drag & drop and enable all hidden button and crop
// فانكشن بتهندل الصور الي جاية من الدراج و دروب و بتشغل الكروب و هكذا
function handleFiles(files) {
    const file = files[0];
    const reader = new FileReader();
    reader.onload = () => {
        const image = new Image();
        image.src = reader.result;
        const imageDiv = document.createElement('div');
        imageDiv.classList.add('image-preview-item', 'image-crop');
        imageDiv.appendChild(image);
        imagePreview.innerHTML = "";
        imagePreview.appendChild(imageDiv);
        const image2 = new Image();
        image2.src = reader.result;
        const imageDiv2 = document.createElement('div');
        imageDiv2.classList.add('image-preview-item2', 'image-crop');
        imageDiv2.appendChild(image2);
        imagePreview2.innerHTML = "";
        imagePreview2.appendChild(imageDiv2);
        showCropButton();
        showCropButton2();
        let croppedImageData = null;
        let croppedImageData2 = null;
        // Add click event listener to start cropping process
        imageDiv.addEventListener('click', () => {
            const cropper = new Cropper(image, {
                // aspectRatio: {
                //     min: 4 / 3,
                //     max: 16 / 9
                // },
                // dragMode: 'move',
                // cropBoxResizable: true,
                // cropBoxMovable: true,
                // toggleDragModeOnDblclick: false,

                // set the aspect ratio of the crop box
                crop: (event) => {

                    // do something with the crop data (e.g. save it to a hidden input field)
                    console.log(event.detail.x, event.detail.y, event.detail.width, event.detail.height);
                    croppedImageData = cropper.getCroppedCanvas().toDataURL();
                },
                ready: () => {
                    // show the crop box once the cropper is ready
                    cropper.setCropBoxData({ width: image.naturalWidth, height: image.naturalHeight });
                }
            });
        });
        imageDiv2.addEventListener('click', () => {
            const cropper2 = new Cropper(image2, {
                crop: (event) => {
                    // do something with the crop data (e.g. save it to a hidden input field)
                    console.log(event.detail.x, event.detail.y, event.detail.width, event.detail.height);
                    croppedImageData2 = cropper2.getCroppedCanvas().toDataURL();
                },
                ready: () => {
                    // show the crop box once the cropper is ready
                    cropper2.setCropBoxData({ width: image2.naturalWidth, height: image2.naturalHeight });
                }
            });
        });
        const cropButton = document.getElementById('crop-button');
        const errorCROP = document.getElementById("crop-error");
        // const splitButton = document.getElementById('split-button');
        cropButton.addEventListener('click', () => {
            if (croppedImageData) {
                errorCROP.style.display = "none"
                // splitButton.style.display = 'block';
                const croppedImage = new Image();
                croppedImage.src = croppedImageData;
                const croppedImageDiv = document.createElement('div');
                croppedImageDiv.appendChild(croppedImage);
                document.getElementById('cropped-image').innerHTML = "";
                document.getElementById('cropped-image').appendChild(croppedImageDiv);
                // const croppedImage2 = new Image();
                // croppedImage2.src = croppedImageData;
                // const croppedImageDiv2 = document.createElement('div');
                // croppedImageDiv2.appendChild(croppedImage2);
                // document.getElementById('cropped-image2').innerHTML = "";
                // document.getElementById('cropped-image2').appendChild(croppedImageDiv2);
            } else {
                errorCROP.innerText = 'no image selected to crop';
                errorCROP.style.display = 'block';
                setTimeout(function () {
                    errorCROP.style.opacity = 1;
                }, 100);

                setTimeout(function () {
                    errorCROP.style.opacity = 0;
                }, 2000);

                setTimeout(function () {
                    errorCROP.style.display = 'none';
                }, 3000);

            }
        });
        const cropButton2 = document.getElementById('crop-button2');
        const errorCROP2 = document.getElementById("crop-error2");

        cropButton2.addEventListener('click', () => {
            if (croppedImageData2) {
                errorCROP2.style.display = "none"
                directChildren = numberOfCol();
                const croppedImage = new Image();
                croppedImage.src = croppedImageData2;
                const inputDiv = document.createElement('input');
                inputDiv.type = "number";
                inputDiv.value = directChildren + 1;
                inputDiv.classList.add('image-number-input');

                const img = document.createElement('img');
                img.src = croppedImageData2;

                const deleteDiv = document.createElement('div');
                deleteDiv.classList.add('delete-icon');
                var deleteIcons = document.querySelectorAll('.delete-icon');
                const croppedImageDiv = document.createElement('div');


                croppedImageDiv.classList.add('image');
                croppedImageDiv.appendChild(inputDiv);
                croppedImageDiv.appendChild(img);
                croppedImageDiv.appendChild(deleteDiv);
                // document.getElementById('cropped-image').innerHTML = "";
                document.querySelector('.image-container').appendChild(croppedImageDiv);
                // const croppedImage2 = new Image();
                // croppedImage2.src = croppedImageData;
                // const croppedImageDiv2 = document.createElement('div');
                // croppedImageDiv2.appendChild(croppedImage2);
                // document.getElementById('cropped-image2').innerHTML = "";
                // document.getElementById('cropped-image2').appendChild(croppedImageDiv2);
            } else {
                errorCROP2.innerText = 'no image selected to crop';
                errorCROP2.style.display = 'block';
                setTimeout(function () {
                    errorCROP2.style.opacity = 1;
                }, 100);

                setTimeout(function () {
                    errorCROP2.style.opacity = 0;
                }, 2000);

                setTimeout(function () {
                    errorCROP2.style.display = 'none';
                }, 3000);

            }
        });
    };
    reader.readAsDataURL(file);
}


const clear = document.getElementById('clear');
const croppedimage = document.getElementById('cropped-image');
const saveButton = document.getElementById('crop-button');
const cropDescription = document.getElementById('crop-description');

// Add this EventListener to add some style when click to crop
// مجرد بيخغي جملة ال click to crop
imagePreview.addEventListener('click', () => {
    const cropDescription = document.getElementById('crop-description');
    const imagePreiewItem = document.querySelector('.image-preview-item');

    cropDescription.style.display = "none";
    imagePreiewItem.style.opacity = 1;

});


imagePreview2.addEventListener('click', () => {
    const cropDescription = document.getElementById('crop-description2');
    const imagePreiewItem2 = document.querySelector('.image-preview-item2');

    cropDescription.style.display = "none";
    imagePreiewItem2.style.opacity = 1;

});





clear.addEventListener('click', () => {
    fileInput.value = "";
    imagePreview.innerHTML = "";
    croppedimage.innerHTML = "";
    clear.style.display = "none";
    saveButton.style.display = "none";
    cropDescription.style.display = "none";

});
// clear2.addEventListener('click', () => {
//     fileInput.value = "";
//     imagePreview2.innerHTML = "";
//     // croppedimage.innerHTML = "";
//     clear2.style.display = "none";
//     saveButton2.style.display = "none";
//     cropDescription2.style.display = "none";
// });



$(document).ready(function () {
    var button = document.getElementById('next-btn');
    var currentDivIndex = 0;
    var divs = $('.div');
    var numDivs = divs.length;
    button.classList.add('api-1');
    // Hide back button initially
    $('.back-btn').prop('disabled', true);

    // Check if Next button should be enabled
    function checkNextEnabled() {
        if (currentDivIndex < numDivs - 1) {
            $('.next-btn').addClass('enabled');
            $('.next-btn').prop('disabled', false);
        } else {
            $('.next-btn').removeClass('enabled');
            $('.next-btn').prop('disabled', true);
        }
    }

    // Check if Back button should be enabled
    function checkBackEnabled() {
        if (currentDivIndex > 0) {
            $('.back-btn').addClass('enabled');
            $('.back-btn').prop('disabled', false);

        } else {
            $('.back-btn').removeClass('enabled');
            $('.back-btn').prop('disabled', true);
        }
    }

    // Show current div and hide all others
    function showCurrentDiv() {
        divs.addClass('hidden');
        $(divs[currentDivIndex]).removeClass('hidden');
    }

    // When Next button is clicked
    $('.next-btn').click(function () {
        if (currentDivIndex < numDivs - 1) {
            currentDivIndex++;
            showCurrentDiv();
            checkBackEnabled();

            //your code here

            // if (currentDivIndex == 0) {
            //     button.setAttribute("id", "next-btn")
            // }
            // else if (currentDivIndex == 1) {
            //     button.setAttribute("id", "next-btn-2")

            // }
            // else if (currentDivIndex == 2) {
            //     button.classList.add('api-3');
            //     button.classList.remove('api-1');
            //     button.classList.remove('api-2');
            //     button.classList.remove('api-4');
            // }
            // else if (currentDivIndex == 3) {
            //     button.classList.add('api-4');
            //     button.classList.remove('api-1');
            //     button.classList.remove('api-2');
            //     button.classList.remove('api-3');

            // }

        }
        checkNextEnabled();
    });


    // $('#split-button').click(function () {
    //     if (currentDivIndex < numDivs - 1) {
    //         currentDivIndex++;
    //         showCurrentDiv();
    //         checkBackEnabled();
    //     }
    //     checkNextEnabled();
    // });
    // When Back button is clicked
    $('.back-btn').click(function () {
        if (currentDivIndex > 0) {
            currentDivIndex--;
            showCurrentDiv();
            checkNextEnabled();

            //your code here

            // if (currentDivIndex == 0) {
            //     button.classList.add('api-1');
            //     button.classList.remove('api-2');
            //     button.classList.remove('api-3');
            //     button.classList.remove('api-4');
            // }
            // else if (currentDivIndex == 1) {
            //     button.classList.add('api-2');
            //     button.classList.remove('api-1');
            //     button.classList.remove('api-3');
            //     button.classList.remove('api-4');
            // }
            // else if (currentDivIndex == 2) {
            //     button.classList.add('api-3');
            //     button.classList.remove('api-1');
            //     button.classList.remove('api-2');
            //     button.classList.remove('api-4');
            // }
            // else if (currentDivIndex == 3) {
            //     button.classList.add('api-4');
            //     button.classList.remove('api-1');
            //     button.classList.remove('api-2');
            //     button.classList.remove('api-3');

            // }

        }
        checkBackEnabled();
    });

    // Initial checks
    checkNextEnabled();

});

// const rowContainer = document.getElementById("row-container");
// const image = document.getElementById("image");
// const canvas = document.getElementById("canvas");
// const ctx = canvas.getContext("2d");

// let isDrawing = false;
// let startX;
// let startY;
// let currentX;
// let currentY;
// const boxes = [];

// Add a mousedown event listener to the image container
// rowContainer.addEventListener("mousedown", (event) => {
//     isDrawing = true;
//     startX = event.clientX - image.offsetLeft;
//     startY = event.clientY - image.offsetTop;
//     currentX = startX;
//     currentY = startY;
// });

// Add a mousemove event listener to the image container
// rowContainer.addEventListener("mousemove", (event) => {
//     if (isDrawing) {
//         currentX = event.clientX - image.offsetLeft;
//         currentY = event.clientY - image.offsetTop;

//         canvas.width = image.width;
//         canvas.height = image.height;

//         ctx.drawImage(image, 0, 0);

//         boxes.forEach((box) => {
//             ctx.strokeStyle = "red";
//             ctx.lineWidth = 2;
//             ctx.strokeRect(box.left, box.top, box.width, box.height);
//         });

//         ctx.strokeStyle = "red";
//         ctx.lineWidth = 2;
//         ctx.strokeRect(
//             Math.min(currentX, startX),
//             Math.min(currentY, startY),
//             Math.abs(currentX - startX),
//             Math.abs(currentY - startY)
//         );
//     }
// });

// Add a mouseup event listener to the image container
// rowContainer.addEventListener("mouseup", () => {
//     if (isDrawing) {
//         isDrawing = false;

//         const width = Math.abs(currentX - startX);
//         const height = Math.abs(currentY - startY);

//         const left = Math.min(currentX, startX);
//         const top = Math.min(currentY, startY);

//         boxes.push({ left, top, width, height });

//         ctx.strokeStyle = "red";
//         ctx.lineWidth = 2;
//         ctx.strokeRect(left, top, width, height);
//     }
// });

// Add a function to draw all boxes onto the image
// function drawBoxesOnImage() {
//     boxes.forEach((box) => {
//         ctx.strokeStyle = "red";
//         ctx.lineWidth = 2;
//         ctx.strokeRect(box.left, box.top, box.width, box.height);
//     });
// }

// // Add a function to save the image with the boxes
// function saveImage() {
//     drawBoxesOnImage();
//     const link = document.createElement("a");
//     link.download = "image_with_boxes.png";
//     link.href = canvas.toDataURL();
//     link.click();
// }

// // Add a function to delete the last box from the list
// function deleteLastBox() {
//     boxes.pop();
//     canvas.width = image.width;
//     canvas.height = image.height;
//     ctx.drawImage(image, 0, 0);
//     drawBoxesOnImage();
// }


function checkDirection() {
    var selectElement = document.getElementById("direction");
    var imageContainer2 = document.querySelector(".image-container");
    var imgElements = imageContainer2.querySelectorAll("img");
    if (selectElement.value === "horizontal") {
        imgElements.forEach(function (imgElement) {
            imgElement.style.width = "400px";
            imgElement.style.height = "100px";
            if (window.innerWidth <= 650) {
                imgElement.style.width = "240px";
                imgElement.style.height = "50px";
            }
        });
    }

}
setInterval(checkDirection, 100);
function googleTranslateElementInit() {
    new google.translate.TranslateElement({
        pageLanguage: 'en',

        layout: google.translate.TranslateElement.InlineLayout.SIMPLE,
        // add the notranslate class to the body element
        // to prevent it from being translated
        excludedClasses: "notranslate"
    }, 'google_translate_element');
}

function translateMyDiv() {
    // get the text to translate from the div
    var textToTranslate = document.getElementById('google-translate').textContent;
    // get the Google Translate API instance
    var googleTranslateApi = new google.translate.TranslateElement({ autoDisplay: false });
    // use the Google Translate API to translate the text
    googleTranslateApi.translatePage([textToTranslate]);
    // display the translated text in the div
    document.getElementById('google-translate').innerHTML = googleTranslateApi.getTranslatedText();
}



const hintButtons = document.querySelectorAll('.hint-button');
hintButtons.forEach(button => {
    const hintText = button.nextElementSibling;
    const hintClose = hintText.querySelector('.hint-close');
    button.addEventListener('click', () => {
        hintText.classList.toggle('active');
    });
    hintClose.addEventListener('click', () => {
        hintText.classList.remove('active');
    });
});
// const cameraContainer = document.getElementById('camera');
// const activateCameraButton = document.getElementById('start-camera');
// navigator.mediaDevices.getUserMedia({ video: true })
//     .then((stream) => {
//         video.srcObject = stream;
//     });
// captureButton.addEventListener('click', () => {
//     const canvas = document.createElement('canvas');
//     canvas.width = video.videoWidth;
//     canvas.height = video.videoHeight;
//     const context = canvas.getContext('2d');
//     context.drawImage(video, 0, 0, canvas.width, canvas.height);
//     const dataURL = canvas.toDataURL('image/png');
//     const image = new Image();
//     image.src = dataURL;
//     const imageDiv = document.createElement('div');
//     imageDiv.classList.add('image-preview-item');
//     imageDiv.appendChild(image);
//     imagePreview.appendChild(imageDiv);
// });
// function handleFiles(files) {
//     for (let i = 0; i < files.length; i++) {
//         const file = files[i];
//         const reader = new FileReader();
//         reader.onload = () => {
//             const image = new Image();
//             image.src = reader.result;
//             const imageDiv = document.createElement('div');
//             imageDiv.classList.add('image-preview-item');
//             imageDiv.appendChild(image);
//             imagePreview.appendChild(imageDiv);
//         };
//         reader.readAsDataURL(file);
//     }
// }
// function handleFiles(files) {
//     const file = files[0];
//     const reader = new FileReader();
//     reader.onload = () => {
//         const image = new Image();
//         image.src = reader.result;
//         const imageDiv = document.createElement('div');
//         imageDiv.classList.add('image-preview-item');
//         imageDiv.appendChild(image);
//         imagePreview.innerHTML = "";
//         imagePreview.appendChild(imageDiv);
//     };
//     reader.readAsDataURL(file);
// }

function copyText() {
    var firstBoxText = document.getElementById("firstBox").textContent;
    document.getElementById("secondBox").textContent = firstBoxText;
}

// Select the specific area you're interested in
const sidebar = document.querySelector('.sidebar');
const openButton = document.getElementById("btn");
const closeButton = document.getElementById("cancel");
const accountMenu = document.querySelector('.account-menu');
const userImage = document.querySelector('.user-image');
document.addEventListener('click', (event) => {
    // Check if the clicked element is within the specific area
    if (!sidebar.contains(event.target) && event.target !== openButton) {
        // Do something when the click is outside the specific area
        sidebar.classList.remove('open');
    }
    if (!userImage.contains(event.target) || accountMenu.contains(event.target)) {
        accountMenu.classList.remove('show');
    }

});

openButton.addEventListener('click', () => {
    sidebar.classList.toggle('open');
    // sidebar.classList.remove('close');
});


closeButton.addEventListener('click', () => {
    // sidebar.classList.toggle('close');
    sidebar.classList.remove('open');
});

userImage.addEventListener('click', () => {
    accountMenu.classList.toggle('show');
});



// const splitButton = document.getElementById('split-button');
// splitButton.addEventListener('click', () => {
//     const cutContainer = document.querySelector('.cut-container');

//     cutContainer.style.display = "flex";
//     cutContainer.scrollIntoView({ behavior: 'smooth' });
// });
const directionSelect = document.getElementById('direction');
const quantityLabel = document.getElementById('quantity-label');

directionSelect.addEventListener('change', function () {
    const selectedValue = directionSelect.value;

    if (selectedValue === 'horizontal') {
        quantityLabel.textContent = 'Number of rows:';
    } else if (selectedValue === 'vertical') {
        quantityLabel.textContent = 'Number of columns:';
    }
});
// Get the image element and the integer variable from the HTML page
var btn = document.getElementById('next-btn');
btn.addEventListener('click', () => {
    // const imageContainer = document.getElementById('image-container');
    document.querySelector('.image-container').innerHTML = "";
    const api_url = "http://127.0.0.1:5000/handle_image_and_integer_and_path";
    const path = 'D:/GP/houghlineoutput';
    const quantityInput = document.querySelector('input[name="quantity"]');

    // get the value of the input
    const quantityValue = quantityInput.value;
    const x = quantityValue;
    const directionSelect = document.getElementById('direction');
    const selectedDirection = directionSelect.value;
    console.log(selectedDirection)
    // get the selected value

    // Get the image from the <img> tag
    const img = document.querySelector('#cropped-image img');

    // Create a canvas element and draw the image onto it
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    // Get the binary string representation of the image data
    const image_data_url = canvas.toDataURL('image/jpeg');
    const image_data_binary = atob(image_data_url.split(',')[1]);

    // Convert the binary string to a Uint8Array
    const image_data_array = new Uint8Array(image_data_binary.length);
    for (let i = 0; i < image_data_binary.length; i++) {
        image_data_array[i] = image_data_binary.charCodeAt(i);
    }

    // Send the image data, file path, and integer variable to the Flask API
    const formData = new FormData();
    formData.append('x', x);
    formData.append('path', path);
    formData.append('image', new Blob([image_data_array], { type: 'image/jpeg' }));
    formData.append('direction', selectedDirection)
    fetch(api_url, {
        method: 'POST',
        body: formData
    }).then(response => response.json())
        .then(data => {
            console.log("thedata" + data);
            data.images.forEach(encodedImage => {
                const imgElement = document.createElement('img');
                imgElement.src = `data:images/jpeg;base64,${encodedImage}`;

                imageContainer.appendChild(imgElement);
                const inputDiv11 = document.createElement('input');
                directChildren = numberOfCol();
                inputDiv11.type = "number";
                inputDiv11.value = directChildren;
                inputDiv11.classList.add('image-number-input');



                const deleteDiv11 = document.createElement('div');
                deleteDiv11.classList.add('delete-icon');

                const croppedImageDiv11 = document.createElement('div');


                croppedImageDiv11.classList.add('image');
                croppedImageDiv11.appendChild(inputDiv11);
                croppedImageDiv11.appendChild(imgElement);
                croppedImageDiv11.appendChild(deleteDiv11);
                // const imgSize = document.createElement('div');
                // imgSize.classList.add("imgSize");
                // imgSize.appendChild(croppedImageDiv11);

                document.querySelector('.image-container').appendChild(croppedImageDiv11);


            });
        })
        .catch(error => console.error(error));
});
function showLoadingScreen() {
    document.getElementById("loading-overlay").classList.remove("hide");
}

// Hide the "Please wait" pop-up screen
function hideLoadingScreen() {
    document.getElementById("loading-overlay").classList.add("hide");
}
document.getElementById("next-btn-2").addEventListener("click", function () {
    showLoadingScreen();
    // Get the image sources
    var imageTags = document.getElementsByClassName("image");
    var imageFiles = [];

    for (var i = 0; i < imageTags.length; i++) {
        var img = imageTags[i].getElementsByTagName("img")[0];
        var src = img.getAttribute("src");
        var filename = src.substring(src.lastIndexOf("/") + 1);
        fetch(src)
            .then(response => response.blob())
            .then(blob => {
                // Create a new file from the blob
                var file = new File([blob], filename, { type: blob.type });
                imageFiles.push(file);

                // Check if all images have been processed
                if (imageFiles.length === imageTags.length) {
                    // Create FormData and append the image files
                    var formData = new FormData();

                    for (var j = 0; j < imageFiles.length; j++) {
                        formData.append("images", imageFiles[j]);
                    }

                    // Send the FormData to the Flask API
                    fetch("http://127.0.0.1:5000/process_images", {
                        method: "POST",
                        body: formData
                    })
                        .then(response => { return response.json() })
                        .then(data => {
                            // Convert the list to a string with new line separation
                            hideLoadingScreen();
                            const stringData = data.join("\n");

                            // Get the target div element
                            const targetDiv = document.getElementById("firstBox");

                            // Create a pre element
                            const pre = document.createElement("pre");

                            // Set the pre element's text content to the string data
                            pre.textContent = stringData;

                            // Append the pre element to the target div
                            targetDiv.innerHTML = "";
                            targetDiv.appendChild(pre);
                        })
                        .catch(error => {
                            console.error(error);
                            hideLoadingScreen();
                        });
                }
            })
            .catch(error => {
                console.error(error);
            });
    }
});

document.getElementById("back-btn").addEventListener('click', () => {
    showLoadingScreen();
})