APP_CSS = """
    Screen { background: #f4efe1; color: #1c1b19; }
    Header { background: #8b0000; color: #fef7e6; }
    Footer { background: #2b3a42; color: #fef7e6; }
    #main-body { height: 1fr; }
    #top-panel {
        height: 30%;
        min-height: 10;
        background: #eee8d5;
        border-bottom: solid #d4af37;
        padding: 0 1;
    }
    #top-left {
        width: 1fr;
        margin-right: 1;
        height: 1fr;
    }
    #top-right {
        width: 2fr;
        height: 1fr;
    }
    #mode-stack { height: auto; }
    .field-card {
        background: #f7f1e2;
        border: round #a33838;
        padding: 0;
        margin-bottom: 0;
    }
    .field-label {
        height: 1;
        margin-left: 0;
        color: #6d5a3d;
    }
    #mode-field, #model-field { width: 1fr; }
    #mode-field {
        height: auto;
        margin-top: 1;
        margin-bottom: 0;
        background: #7f1010;
        border: round #d4af37;
        padding: 0 1;
    }
    #mode-field .field-label {
        color: #f7e7bf;
        text-style: bold;
    }
    #model-field {
        height: auto;
        border: none;
        background: transparent;
        padding: 0;
    }
    #model-field .field-label { color: #6d5a3d; }
    #model-field #model-input {
        background: #fff8ec;
    }
    #file-field {
        height: auto;
        margin-bottom: 0;
    }
    #query-field {
        height: auto;
        min-height: 4;
        background: #f9f2df;
    }
    #query-field .field-label {
        color: #6d5a3d;
        text-style: none;
    }
    #file-input, #search-input, #mode-select, #model-input { width: 1fr; }
    Input {
        height: 1;
        margin: 0;
        padding: 0;
        background: #fffaf0;
        color: #1a1a1a;
        border: none;
    }
    Input:hover { border: none; background: #fff8ec; }
    Input:focus {
        border: none;
        background: #fffdf7;
    }
    #search-input {
        border: none;
        background: #fffdf4;
    }
    #search-input:focus {
        border: none;
        background: #fffef8;
    }
    Select {
        height: auto;
        margin: 0;
        background: transparent;
        color: #1a1a1a;
        border: none;
    }
    Select:hover, Select:focus {
        border: none;
        background: transparent;
    }
    #mode-select {
        height: 3;
        margin: 0;
        padding: 0;
        background: transparent;
        color: #1a1a1a;
        border: none;
    }
    #mode-select > SelectCurrent {
        height: 3;
        margin: 0;
        background: #fff6e2;
        color: #7f1010;
        border: round #d4af37;
        padding: 0 1;
    }
    #mode-select > SelectCurrent:ansi {
        height: 3;
        background: #fff6e2;
        color: #7f1010;
        border: round #d4af37;
    }
    #mode-select > SelectCurrent:hover {
        background: #fff9eb;
    }
    #mode-select:focus > SelectCurrent {
        background: #fffdf7;
        border: round #f0d68a;
    }
    #mode-select > SelectCurrent Static#label {
        color: #7f1010;
        background: transparent;
        text-style: bold;
    }
    #mode-select > SelectCurrent.-has-value Static#label {
        color: #7f1010;
        text-style: bold;
    }
    #mode-select > SelectCurrent .arrow {
        color: #8b5f1a;
        background: transparent;
    }
    #mode-select > SelectOverlay {
        background: #fffaf0;
        color: #1a1a1a;
        border: round #8f2626;
    }
    #mode-select > SelectOverlay:focus {
        border: round #d4af37;
        background: #fffdf7;
    }
    #mode-select > SelectOverlay > .option-list--option {
        color: #1a1a1a;
        background: #fffaf0;
    }
    #mode-select > SelectOverlay > .option-list--option-hover {
        color: #1a1a1a;
        background: #f5ebd3;
    }
    #mode-select > SelectOverlay > .option-list--option-highlighted {
        color: #7f1010;
        background: #f3e6c6;
    }
    #model-field.model-hidden {
        display: none;
    }
    #output-panel { height: 2fr; }
    #sidebar {
        width: 38%;
        background: #2b3a42;
        color: #eee8d5;
        border-right: solid #d4af37;
        padding: 1 0;
    }
    #sidebar, #reader-container {
        scrollbar-size-vertical: 1;
        scrollbar-color: #d8c8a3;
        scrollbar-color-hover: #cab588;
        scrollbar-color-active: #bba06e;
        scrollbar-background: #f6f1e4;
        scrollbar-background-hover: #f0e9d9;
        scrollbar-background-active: #e8ddc4;
        scrollbar-corner-color: #f6f1e4;
    }
    ListItem {
        height: 4;
        margin: 0 1 1 1;
        padding: 0 1;
        border: round #465964;
        background: #31424f;
        color: #f1ecdf;
    }
    ListItem.--highlight, ListItem:hover {
        border: round #d4af37;
        background: #6e1313;
        color: #fff8e7;
    }
    #reader-container { width: 62%; padding: 2 3; }
    #reader { height: auto; }
    """
