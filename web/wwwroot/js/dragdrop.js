window.deceptionLensDragDrop = window.deceptionLensDragDrop || (function () {
    const boundPairs = new Set();

    function bind(uploadZoneId, inputId) {
        const key = uploadZoneId + "::" + inputId;
        if (boundPairs.has(key)) {
            return;
        }

        const zone = document.getElementById(uploadZoneId);
        const input = document.getElementById(inputId);
        if (!zone || !input) {
            return;
        }

        const addDragging = function () {
            zone.classList.add("dragging");
        };

        const removeDragging = function () {
            zone.classList.remove("dragging");
        };

        const prevent = function (event) {
            event.preventDefault();
            event.stopPropagation();
        };

        zone.addEventListener("dragenter", function (event) {
            prevent(event);
            addDragging();
        });

        zone.addEventListener("dragover", function (event) {
            prevent(event);
            addDragging();
        });

        zone.addEventListener("dragleave", function (event) {
            prevent(event);
            removeDragging();
        });

        zone.addEventListener("drop", function (event) {
            prevent(event);
            removeDragging();

            const files = event.dataTransfer && event.dataTransfer.files;
            if (!files || files.length === 0) {
                return;
            }

            const transfer = new DataTransfer();
            transfer.items.add(files[0]);
            input.files = transfer.files;
            input.dispatchEvent(new Event("change", { bubbles: true }));
        });

        boundPairs.add(key);
    }

    return { bind };
})();
