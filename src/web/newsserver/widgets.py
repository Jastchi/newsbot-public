"""Custom Django form widgets for the newsserver admin."""

from collections.abc import Mapping
from typing import Any

from django import forms
from django.forms.renderers import BaseRenderer
from django.utils.datastructures import MultiValueDict
from django.utils.html import format_html
from django.utils.safestring import SafeString

_COLOR_INPUT_STYLE = (
    "width:60px; height:36px; padding:2px; cursor:pointer;"
)


class OptionalColorWidget(forms.Widget):
    """Optional color picker with an enable/disable checkbox."""

    _DEFAULT_COLOR = "#7060da"

    def render(
        self,
        name: str,
        value: Any,
        attrs: dict[str, Any] | None = None,
        renderer: BaseRenderer | None = None,
    ) -> SafeString:
        """Render hidden field, checkbox, and color input."""
        del renderer
        final_attrs = self.build_attrs(self.attrs, attrs or {})
        eid = final_attrs.get("id", name)
        str_value = "" if value is None else str(value)
        has_value = bool(str_value)
        color_val = str_value if has_value else self._DEFAULT_COLOR
        checked = "checked" if has_value else ""
        disabled = "" if has_value else "disabled"
        return format_html(
            '<input type="hidden" name="{}" id="{}_hidden" value="{}">'
            '<label style="display:inline-flex;align-items:center;'
            'gap:8px;cursor:pointer;">'
            '<input type="checkbox" id="{}_enable" {} '
            'onchange="(function(cb){{'
            "var h=document.getElementById('{}_hidden'),"
            "p=document.getElementById('{}_picker');"
            "if(cb.checked){{h.value=p.value;p.disabled=false;}}"
            "else{{h.value='';p.disabled=true;}}"
            "h.dispatchEvent(new Event('change'));"
            '}})(this)">'
            '<input type="color" id="{}_picker" value="{}" {} '
            'style="{}" '
            'oninput="(function(p){{'
            "var e=document.getElementById('{}_enable'),"
            "h=document.getElementById('{}_hidden');"
            "if(e.checked){{h.value=p.value;"
            "h.dispatchEvent(new Event('change'));}}"
            '}})(this)">'
            "</label>",
            name,
            eid,
            str_value,
            eid,
            checked,
            eid,
            eid,
            eid,
            color_val,
            disabled,
            _COLOR_INPUT_STYLE,
            eid,
            eid,
        )

    def value_from_datadict(
        self,
        data: Mapping[str, Any],
        files: MultiValueDict[str, Any],
        name: str,
    ) -> Any:
        """Read the hidden input value submitted with the form."""
        del files
        return data.get(name, "")
