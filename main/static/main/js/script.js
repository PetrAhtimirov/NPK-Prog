let imgArr = document.querySelectorAll("img");
let groupEng = document.querySelector(".groupEng");
let static = document.querySelector(".static");
for (let i = 0; i <= imgArr.length; i++) {
    imgArr[i].src = static.textContent + groupEng.textContent + "/" + (i+1) + ".jpg";
}