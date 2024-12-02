
function toggleLogo(event) {
    var elements = document.querySelectorAll(".slide-logo");
    if (event.currentSlide.matches('#title-slide') || event.currentSlide.classList.contains('no-logo')) {
        // Hide the logo on the title slide
        elements.forEach(function(elem) {
            elem.style.display = "none";
        });
    } else {
        // Show and position the logo on other slides
        elements.forEach(function(elem) {
            elem.style.display = "block";
        });
    }
}

function toggleFooter(event) {
    var elements = document.querySelectorAll("div.footer-default");
    if (event.currentSlide.matches('#title-slide') || event.currentSlide.classList.contains('no-footer')) {
        // Hide the logo on the title slide
        elements.forEach(function(elem) {
            elem.style.display = "none";
        });
    } else {
        // Show and position the logo on other slides
        elements.forEach(function(elem) {
            elem.style.display = "block";
        });
    }
}

window.addEventListener("load", (event) => {
    var elements = document.querySelectorAll(".slide-logo");
    elements.forEach(function(elem) {
        elem.style.display = "none"; // Hide initially
    });
    
    var elements = document.querySelectorAll("div.footer-default");
    elements.forEach(function(elem) {
        elem.style.display = "none"; // Hide initially
    });

    Reveal.on("slidechanged", function(event) {
        toggleLogo(event);
        toggleFooter(event);
    });
});


