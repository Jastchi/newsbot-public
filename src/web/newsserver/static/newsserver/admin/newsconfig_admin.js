(function () {
    "use strict";

    var debounceTimer = null;

    function updatePreview() {
        var iframe = document.getElementById("email-preview-iframe");
        if (!iframe) return;

        var primaryInput = document.getElementById("id_hero_color_primary");
        var secondaryInput = document.getElementById("id_hero_color_secondary");
        var middleInput = document.getElementById("id_hero_color_middle_hidden");
        if (!primaryInput || !secondaryInput) return;

        var primary = primaryInput.value.trim();
        var secondary = secondaryInput.value.trim();
        if (!primary || !secondary) return;

        var base = iframe.src.split("?")[0];
        var url =
            base +
            "?primary=" +
            encodeURIComponent(primary) +
            "&secondary=" +
            encodeURIComponent(secondary);

        var middle = middleInput ? middleInput.value.trim() : "";
        if (middle) {
            url += "&middle=" + encodeURIComponent(middle);
        }

        iframe.src = url;
    }

    function scheduleUpdate() {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(updatePreview, 300);
    }

    function layoutEmailPreviewSection() {
        var primaryRow = document.querySelector(".field-hero_color_primary");
        var secondaryRow = document.querySelector(".field-hero_color_secondary");
        var middleRow = document.querySelector(".field-hero_color_middle");
        var previewRow = document.querySelector(".field-email_preview_panel");

        if (!primaryRow || !secondaryRow || !previewRow) return;

        var fieldset = primaryRow.closest("fieldset");
        if (!fieldset || fieldset !== previewRow.closest("fieldset")) return;

        var wrapper = document.createElement("div");
        wrapper.style.cssText = "display:flex; gap:24px; align-items:flex-start;";

        var leftCol = document.createElement("div");
        leftCol.style.cssText = "min-width:240px; flex-shrink:0;";

        var rightCol = document.createElement("div");
        rightCol.style.cssText = "flex:1; min-width:0;";

        fieldset.insertBefore(wrapper, primaryRow);
        leftCol.appendChild(primaryRow);
        leftCol.appendChild(secondaryRow);
        if (middleRow) leftCol.appendChild(middleRow);
        rightCol.appendChild(previewRow);
        wrapper.appendChild(leftCol);
        wrapper.appendChild(rightCol);
    }

    document.addEventListener("DOMContentLoaded", function () {
        var primaryInput = document.getElementById("id_hero_color_primary");
        var secondaryInput = document.getElementById("id_hero_color_secondary");
        var middleHidden = document.getElementById("id_hero_color_middle_hidden");

        if (primaryInput) {
            primaryInput.addEventListener("input", scheduleUpdate);
            primaryInput.addEventListener("change", scheduleUpdate);
        }
        if (secondaryInput) {
            secondaryInput.addEventListener("input", scheduleUpdate);
            secondaryInput.addEventListener("change", scheduleUpdate);
        }
        if (middleHidden) {
            middleHidden.addEventListener("change", scheduleUpdate);
        }

        layoutEmailPreviewSection();
    });
})();
