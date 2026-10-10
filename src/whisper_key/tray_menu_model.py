from dataclasses import dataclass, field
from typing import Callable, Optional, Union

BoolOrGetter = Union[bool, Callable[[], bool]]


def resolve(value):
    return value() if callable(value) else value


@dataclass
class Header:
    title: str
    subtitle: Callable[[], str]
    muted: Callable[[], bool]
    on_toggle_mute: Callable[[], None]


@dataclass
class Section:
    label: str


@dataclass
class Info:
    label: str
    value: Callable[[], str]


@dataclass
class Action:
    label: str
    on_select: Callable[[], None]
    icon: Optional[str] = None
    enabled: BoolOrGetter = True
    default: bool = False


@dataclass
class Toggle:
    label: str
    is_on: Callable[[], bool]
    on_change: Callable[[bool], None]
    enabled: BoolOrGetter = True
    hint: Optional[str] = None


@dataclass
class Option:
    value: object
    label: str
    enabled: bool = True
    starts_group: bool = False


@dataclass
class Choice:
    label: str
    options: Union[list, Callable[[], list]]
    current: Callable[[], object]
    on_select: Callable[[object], None]
    style: str = "list"
    preview: Optional[Callable[[object], object]] = None
    preview_key: Optional[Callable[[], object]] = None

    def resolved_options(self) -> list:
        return resolve(self.options)


@dataclass
class Slider:
    label: str
    minimum: float
    maximum: float
    step: float
    current: Callable[[], float]
    on_change: Callable[[float], None]
    on_commit: Callable[[float], None]
    format_value: Callable[[float], str]
    presets: list = field(default_factory=list)
    track_colors: Optional[Callable[[], list]] = None
    default_value: Optional[float] = None
    snap_points: list = field(default_factory=list)
    snap_distance: float = 0.0

    def snap(self, value: float) -> float:
        nearest_point = min(self.snap_points, key=lambda point: abs(point - value), default=None)
        if nearest_point is not None and abs(nearest_point - value) <= self.snap_distance:
            return nearest_point
        return self.snap_to_step(value)

    def snap_to_step(self, value: float) -> float:
        steps = round((value - self.minimum) / self.step)
        return round(max(self.minimum, min(self.maximum, self.minimum + steps * self.step)), 6)


@dataclass
class Submenu:
    label: str
    items: Union[list, Callable[[], list]]
    icon: Optional[str] = None
    detail: Optional[Callable[[], str]] = None

    def resolved_items(self) -> list:
        return [item for item in resolve(self.items) if item is not None]


@dataclass
class Footer:
    items: list = field(default_factory=list)


def to_pystray(items: list, pystray):
    return pystray.Menu(*_pystray_items(items, pystray))


def _pystray_items(items: list, pystray) -> list:
    out = []

    def separator():
        if out and out[-1] is not pystray.Menu.SEPARATOR:
            out.append(pystray.Menu.SEPARATOR)

    for item in items:
        if item is None:
            continue
        if isinstance(item, Header):
            out.append(pystray.MenuItem("Mute microphone", _on_click(item.on_toggle_mute), checked=_getter(item.muted)))
        elif isinstance(item, Section):
            separator()
        elif isinstance(item, Footer):
            separator()
            out += _pystray_items(item.items, pystray)
        elif isinstance(item, Info):
            out.append(pystray.MenuItem(_info_text(item), None, enabled=False))
        elif isinstance(item, Action):
            out.append(pystray.MenuItem(item.label, _on_click(item.on_select), enabled=_enabled(item.enabled),
                                        default=item.default))
        elif isinstance(item, Toggle):
            out.append(pystray.MenuItem(item.label, _on_click(_flip(item)), checked=_getter(item.is_on),
                                        enabled=_enabled(item.enabled)))
        elif isinstance(item, Choice):
            radios = _pystray_choice(item, pystray)
            if item.style == "segmented":
                out.append(pystray.MenuItem(item.label, pystray.Menu(*radios)))
                continue
            separator()
            if len(items) > 1:
                out.append(pystray.MenuItem(item.label, None, enabled=False))
            out += radios
        elif isinstance(item, Slider):
            out.append(pystray.MenuItem(f"{item.label}: {item.format_value(item.current())}",
                                        pystray.Menu(*_pystray_slider_presets(item, pystray))))
        elif isinstance(item, Submenu):
            detail = item.detail() if item.detail else None
            label = f"{item.label}: {detail}" if detail else item.label
            children = _pystray_items(item.resolved_items(), pystray)
            if children:
                out.append(pystray.MenuItem(label, pystray.Menu(*children)))

    while out and out[-1] is pystray.Menu.SEPARATOR:
        out.pop()
    return out


def _pystray_choice(choice: Choice, pystray) -> list:
    radios = []
    for option in choice.resolved_options():
        if option.starts_group and radios:
            radios.append(pystray.Menu.SEPARATOR)
        radios.append(pystray.MenuItem(option.label, _on_click(_select(choice, option.value)), radio=True,
                                       checked=_getter(_is_current(choice, option.value)), enabled=option.enabled))
    return radios


def _pystray_slider_presets(slider: Slider, pystray) -> list:
    return [pystray.MenuItem(slider.format_value(value), _on_click(_commit(slider, value)), radio=True,
                             checked=_getter(_slider_is_at(slider, value)))
            for value in slider.presets]


def _commit(slider: Slider, value):
    return lambda: slider.on_commit(value)


def _slider_is_at(slider: Slider, value):
    return lambda: abs(slider.current() - value) < slider.step / 2


def _info_text(info: Info):
    return lambda _item: f"{info.label}: {info.value()}"


def _enabled(value):
    return value if isinstance(value, bool) else _getter(value)


def _on_click(callback):
    return lambda icon, menu_item: callback()


def _getter(callback):
    return lambda menu_item: callback()


def _flip(toggle: Toggle):
    return lambda: toggle.on_change(not toggle.is_on())


def _select(choice: Choice, value):
    return lambda: choice.on_select(value)


def _is_current(choice: Choice, value):
    return lambda: choice.current() == value

