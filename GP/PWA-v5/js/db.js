'use strict';
// let rawWords = "fkjfgf";
// let translatedWords = "fsfsff";
var currentUID;
var signInButton = document.getElementById('sign-in-button');
var signOutButton = document.getElementById('sign-out-button');
var welcomeMessage = document.getElementById('welcome-message');
var continueButton = document.getElementById('continue');

function createNewObject(userId, rawWords, translatedWords) {
    // console.log("Hi finally");
    firebase.database().ref('History/' + userId).push({
        rawWords: rawWords,
        translatedWords: translatedWords
    });
    // console.log("Hi finally 2");
}

function fetchObjects(userId) {
    var translatationRef = firebase.database().ref('History/' + userId);

    translatationRef.on('child_added', function (data) {
        onObjectAdded(data);

    });
    translatationRef.on('child_changed', function (data) {
        onObjectAdded(data);
    });
}

//Receive the data from the database.
function onObjectAdded(data) {
    // var author = data.val().author || 'Anonymous';
    var data = data.val() || 'Anonymous';
    console.log(data);
}

// Display the user's data from the database.
function displayUserData(userId, name, email, imageUrl) {
    var user = "Welcome" + ' ' + name;
    // const joined = `${one}${two}`;
    welcomeMessage.innerHTML = user;
}
/**
 * Triggers every time there is a change in the Firebase auth state (i.e. user signed-in or user signed out).
 */
function onAuthStateChanged(user) {
    // We ignore token refresh events.
    if (user && currentUID === user.uid) {
        return;
    }

    if (user) {
        currentUID = user.uid;
        signInButton.style.display = 'none';
        signOutButton.style.display = 'block';
        continueButton.style.display = 'block';
        // window.location.href = "../PWA.html";

        //Link to another page
        displayUserData(user.uid, user.displayName, user.email, user.photoURL);
        fetchObjects(currentUID);

        // window. history. back()

        // createNewObject(currentUID, "Apple","تفاحه");
        // createNewObject(currentUID, "ball", "كرة");
        // createNewObject(currentUID, Apple, تفاحه);
        // createNewObject(currentUID, rawWords, translatedWords);
        // createNewObject(currentUID, rawWords, translatedWords);
    } else {
        // Set currentUID to null.
        currentUID = null;
        // Display the page where you can sign-in.
        displayUserData('', '', '', '');
        signInButton.style.display = 'block';
        signOutButton.style.display = 'none';
        continueButton.style.display = 'none';
        //Link to another page
        // window. history. back()
    }
}

// Bindings on load.
window.addEventListener('load', function () {
    // Bind Sign in button.
    signInButton.addEventListener('click', function () {
        var provider = new firebase.auth.GoogleAuthProvider();
        firebase.auth().signInWithPopup(provider);
    });
    // Bind Sign out button.
    signOutButton.addEventListener('click', function () {
        firebase.auth().signOut();
    });

    // Listen for auth state changes
    firebase.auth().onAuthStateChanged(onAuthStateChanged);
}, false);

// firebase.auth().onAuthStateChanged(user => {
//     if (user) {
//         window.location = 'home.html'; //After successful login, user will be redirected to home.html
//     }
// });


// var fetchPosts = function (postsRef, sectionElement) {
//     postsRef.on('child_added', function (data) {
//         var author = data.val().author || 'Anonymous';
//         var containerElement = sectionElement.getElementsByClassName('posts-container')[0];
//         containerElement.insertBefore(
//             createPostElement(data.key, data.val().title, data.val().body, author, data.val().uid, data.val().authorPic),
//             containerElement.firstChild);
//     });
//     postsRef.on('child_changed', function (data) {
//         var containerElement = sectionElement.getElementsByClassName('posts-container')[0];
//         var postElement = containerElement.getElementsByClassName('post-' + data.key)[0];
//         postElement.getElementsByClassName('mdl-card__title-text')[0].innerText = data.val().title;
//         postElement.getElementsByClassName('username')[0].innerText = data.val().author;
//         postElement.getElementsByClassName('text')[0].innerText = data.val().body;
//         postElement.getElementsByClassName('star-count')[0].innerText = data.val().starCount;
//     });
//     postsRef.on('child_removed', function (data) {
//         var containerElement = sectionElement.getElementsByClassName('posts-container')[0];
//         var post = containerElement.getElementsByClassName('post-' + data.key)[0];
//         post.parentElement.removeChild(post);
//     });
// };