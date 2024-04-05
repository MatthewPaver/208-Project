const apiKey = "0KQVHmGf295NtumVdVX2AESVuugIZJ82nRc8oi6OPG8"

const elementForm = document.querySelector("form");
const elementPrompt = document.getElementById("prompt");
const generatedResults = document.querySelector(".imageDispay");

let dataInput =  " ";


async function generateImages(){

    generatedResults.innerHTML = '';

    dataInput = elementPrompt.value;
    const dynamicURL = `https://api.unsplash.com/search/photos?query=${dataInput}&client_id=${apiKey}`;

    const response = await fetch(dynamicURL);
    const data = await response.json();

    const results = data.results;

    if (data.results.length > 0) {
        const result = data.results[0]; 
        const imageContainer = document.createElement('div');
        imageContainer.classList.add("imageAPI");
        const image = document.createElement('img');
        image.src = result.urls.small;
        image.alt = result.alt_description;
        const imageLink = document.createElement('a');
        imageLink.href = result.links.html;
        imageLink.target = "_blank";
        imageLink.textContent = "view on Unsplash";

        imageContainer.appendChild(image);
        imageContainer.appendChild(imageLink);

        generatedResults.appendChild(imageContainer);

    

         };

        
}

elementForm.addEventListener("submit", (event) =>{

    event.preventDefault();
    generateImages();

}) 


/* NOTE in the video he talked about using a show more pages button. I did not include it because testing needed only one picture.
However, we could use that to add a "edit" feautre where the generateImages() is called with the previous generated image. */


/* Refernce Youtuber channel (Tech2 etc) Video (JavaScript Project | Image Search App With JavaScript API) URL
: https://www.youtube.com/watch?v=oaliV2Dp7WQ */