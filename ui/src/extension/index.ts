import {
  CellClassParams,
  CellClickedEvent,
  createGrid,
  GridApi,
  GridOptions,
  ICellRendererParams,
  ITooltipParams,
  RowHighlightPosition,
  RowNode,
  ValueFormatterParams,
  ValueGetterParams,
} from 'ag-grid-community';
import { Notyf } from 'notyf';

import bookmark from '../assets/icons/bookmark.svg?raw';
import bookmarked from '../assets/icons/bookmark-filled.svg?raw';
import cancelIcon from '../assets/icons/cancel.svg?raw';
import deleteIcon from '../assets/icons/delete.svg?raw';
import playIcon from '../assets/icons/play.svg?raw';
import rotateIcon from '../assets/icons/rotate.svg?raw';
import saveIcon from '../assets/icons/save.svg?raw';
import searchIcon from '../assets/icons/search.svg?raw';
import { getHighlightPosition, getPixelOnRow, getRowNodeAtPixel } from '../utils/ag-grid';
import { debounce } from '../utils/debounce';
import { extractArgs } from '../utils/extract-args';

import { createHistoryTasksStore } from './stores/history.store';
import { createPendingTasksStore } from './stores/pending.store';
import { createSharedStore } from './stores/shared.store';
import { ProgressResponse, ResponseStatus, Task, TaskStatus } from './types';

import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-alpine.css';
import 'notyf/notyf.min.css';
import './index.scss';

let notyf: Notyf | undefined;

declare global {
  let opts: object;
  function gradioApp(): HTMLElement;
  function randomId(): string;
  function origRandomId(): string;
  function get_tab_index(name: string): number;
  function create_submit_args(args: any[]): any[];
  function requestProgress(
    id: string,
    progressContainer: HTMLElement,
    imagesContainer: HTMLElement,
    onDone?: () => void,
    onProgress?: (res: ProgressResponse) => void
  ): void;
  function onUiLoaded(callback: () => void): void;
  function notify(response: ResponseStatus): void;
  function submit(...args: any[]): any[];
  function submit_txt2img(...args: any[]): any[];
  function submit_img2img(...args: any[]): any[];
  function submit_enqueue(...args: any[]): any[];
  function submit_enqueue_img2img(...args: any[]): any[];
  function agent_scheduler_status_filter_changed(value: string): void;
  function appendContextMenuOption(selector: string, label: string, callback: () => void): void;
  function modalSaveImage(event: Event): void;
}

const sharedStore = createSharedStore({
  uiAsTab: true,
  selectedTab: 'pending',
});

const pendingStore = createPendingTasksStore({
  current_task_id: null,
  total_pending_tasks: 0,
  pending_tasks: [],
  paused: false,
});

const historyStore = createHistoryTasksStore({
  total: 0,
  tasks: [],
});

// load samplers and checkpoints
const samplers: string[] = [];
const checkpoints: string[] = ['System'];

const sharedGridOptions: GridOptions<Task> = {
  // default col def properties get applied to all columns
  defaultColDef: {
    sortable: false,
    filter: true,
    resizable: true,
    suppressMenu: true,
  },
  // each entry here represents one column
  columnDefs: [
    {
      field: 'name',
      headerName: 'Task Id',
      cellDataType: 'text',
      minWidth: 240,
      maxWidth: 240,
      pinned: 'left',
      rowDrag: true,
      valueGetter: ({ data }: ValueGetterParams<Task, string>) => data?.name ?? data?.id,
      cellClass: ({ data }: CellClassParams<Task, string>) => {
        if (data == null) return;

        return ['cursor-pointer', `task-${data.status}`];
      },
    },
    {
      field: 'type',
      headerName: 'Type',
      minWidth: 80,
      maxWidth: 80,
      editable: false,
    },
    {
      field: 'editing',
      editable: false,
      hide: true,
    },
    {
      headerName: 'Params',
      children: [
        {
          field: 'params.prompt',
          headerName: 'Prompt',
          cellDataType: 'text',
          minWidth: 200,
          maxWidth: 400,
          autoHeight: true,
          wrapText: true,
          cellClass: 'wrap-cell',
        },
        {
          field: 'params.negative_prompt',
          headerName: 'Negative Prompt',
          cellDataType: 'text',
          minWidth: 200,
          maxWidth: 400,
          autoHeight: true,
          wrapText: true,
          cellClass: 'wrap-cell',
        },
        {
          field: 'params.checkpoint',
          headerName: 'Checkpoint',
          cellDataType: 'text',
          minWidth: 150,
          maxWidth: 300,
          valueFormatter: ({ value }: ValueFormatterParams<Task, string | undefined>) =>
            value ?? 'System',
          cellEditor: 'agSelectCellEditor',
          cellEditorParams: () => ({ values: checkpoints }),
        },
        {
          field: 'params.sampler_name',
          headerName: 'Sampler',
          cellDataType: 'text',
          width: 150,
          minWidth: 150,
          cellEditor: 'agSelectCellEditor',
          cellEditorParams: () => ({ values: samplers }),
        },
        {
          field: 'params.steps',
          headerName: 'Steps',
          cellDataType: 'number',
          minWidth: 80,
          maxWidth: 80,
          filter: 'agNumberColumnFilter',
          cellEditor: 'agNumberCellEditor',
          cellEditorParams: {
            min: 1,
            max: 150,
            precision: 0,
            step: 1,
          },
        },
        {
          field: 'params.cfg_scale',
          headerName: 'CFG Scale',
          cellDataType: 'number',
          width: 100,
          minWidth: 100,
          filter: 'agNumberColumnFilter',
          cellEditor: 'agNumberCellEditor',
          cellEditorParams: {
            min: 1,
            max: 30,
            precision: 1,
            step: 0.5,
          },
        },
        {
          field: 'params.size',
          headerName: 'Size',
          minWidth: 110,
          maxWidth: 110,
          editable: false,
          valueGetter: ({ data }: ValueGetterParams<Task, string | undefined>) => {
            const params = data?.params;
            return params != null ? `${params.width} × ${params.height}` : undefined;
          },
        },
        {
          field: 'params.batch',
          headerName: 'Batching',
          minWidth: 100,
          maxWidth: 100,
          editable: false,
          valueGetter: ({ data }: ValueGetterParams<Task, string>) => {
            const params = data?.params;
            return params != null ? `${params.batch_size} × ${params.n_iter}` : '1 × 1';
          },
        },
      ],
    },
    {
      field: 'created_at',
      headerName: 'Queued At',
      minWidth: 180,
      editable: false,
      valueFormatter: ({ value }: ValueFormatterParams<Task, number>) =>
        value != null ? new Date(value).toLocaleString(document.documentElement.lang) : '',
    },
    {
      field: 'updated_at',
      headerName: 'Updated At',
      minWidth: 180,
      editable: false,
      valueFormatter: ({ value }: ValueFormatterParams<Task, number>) =>
        value != null ? new Date(value).toLocaleString(document.documentElement.lang) : '',
    },
  ],

  getRowId: ({ data }) => data.id,
  rowSelection: 'single', // allow rows to be selected
  animateRows: true, // have rows animate to new positions when sorted
  pagination: true,
  paginationAutoPageSize: true,
  suppressCopyRowsToClipboard: true,
  enableBrowserTooltips: true,
};

function initSearchInput(selector: string) {
  const searchContainer = gradioApp().querySelector<HTMLDivElement>(selector);
  if (searchContainer == null) {
    throw new Error(`Search container '${selector}' not found.`);
  }
  const searchInput = searchContainer.getElementsByTagName('input')[0];
  if (searchInput == null) {
    throw new Error('Search input not found.');
  }
  searchInput.classList.add('ts-search-input');

  const searchIconContainer = document.createElement('div');
  searchIconContainer.className = 'ts-search-icon';
  searchIconContainer.innerHTML = searchIcon;
  searchInput.parentElement!.appendChild(searchIconContainer);

  return searchInput;
}

// function initImport(selector: string) {
//   const importContainer = gradioApp().querySelector<HTMLDivElement>(selector);
//   if (importContainer == null) {
//     throw new Error(`Import container '${selector}' not found.`);
//   }
//   const importInput = importContainer.getElementsByTagName('input')[0];
//   if (importInput == null) {
//     throw new Error('Import input not found.');
//   }
//   return importInput;
// }

async function notify(response: ResponseStatus) {
  if (notyf == null) {
    const Notyf = await import('notyf');
    notyf = new Notyf.Notyf({
      position: { x: 'center', y: 'bottom' },
      duration: 3000,
    });
  }

  if (response.success) {
    notyf.success(response.message);
  } else {
    notyf.error(response.message);
  }
}

window.notify = notify;

function makeTaskId(): string {
  if (typeof randomId === 'function') return randomId();
  return (
    'task(' +
    Math.random().toString(36).slice(2, 7) +
    Math.random().toString(36).slice(2, 7) +
    Math.random().toString(36).slice(2, 7) +
    ')'
  );
}

function showTaskProgress(task_id: string, type: string | undefined, callback: () => void) {
  // delay progress request until the options loaded
  if (Object.keys(opts).length === 0) {
    setTimeout(() => showTaskProgress(task_id, type, callback), 500);
    return;
  }

  const args = extractArgs(requestProgress);

  const gallery = gradioApp().querySelector<HTMLDivElement>(
    '#agent_scheduler_current_task_images'
  );
  if (!gallery) {
    console.warn('[AgentScheduler] progress gallery #agent_scheduler_current_task_images missing');
    callback();
    return;
  }

  const onDone = () => {
    // Pull final images into scheduler + active tab galleries (queued runs
    // don't return through Gradio generate, so previews otherwise vanish).
    void displayTaskResultImages(task_id, type).finally(() => callback());
  };

  // A1111 / Forge Neo: inactivityTimeout as 6th arg (default 40s is too short for queued waits)
  const inactivityTimeout = 600;
  if (args.includes('progressbarContainer') || args.includes('inactivityTimeout')) {
    requestProgress(task_id, gallery, gallery, onDone, undefined, inactivityTimeout);
  } else {
    // Vlad version
    const progressDiv = document.createElement('div');
    progressDiv.className = 'progressDiv';
    gallery.parentElement!.insertBefore(progressDiv, gallery);
    requestProgress(
      task_id,
      gallery,
      gallery,
      () => {
        progressDiv.remove();
        onDone();
      },
      res => {
        const perc = `${Math.round(res.progress * 100.0)}%`;
        const eta = res.paused ? 'Paused' : `ETA: ${Math.round(res.eta)}s`;
        progressDiv.innerText = `${perc} ${eta}`;
        progressDiv.style.background = `linear-gradient(to right, var(--primary-500) 0%, var(--primary-800) ${perc}, var(--neutral-700) ${perc})`;
      }
    );
  }

  // monkey patch randomId to return task_id, then call submit to trigger progress
  // on the open txt2img/img2img tab (same UX as Generate)
  const prevRandomId = window.randomId;
  window.randomId = () => task_id;
  try {
    if (type === 'txt2img') {
      submit();
    } else if (type === 'img2img') {
      submit_img2img();
    }
  } catch (e) {
    console.warn('[AgentScheduler] submit() progress hook failed', e);
  }
  window.randomId = prevRandomId;
}

async function displayTaskResultImages(taskId: string, type: string | undefined) {
  try {
    const res = await fetch(`/agent-scheduler/v1/task/${encodeURIComponent(taskId)}/results`);
    const body = await res.json();
    if (!body?.success || !Array.isArray(body.data) || body.data.length === 0) return;

    const imageUrls: string[] = body.data
      .map((item: { image?: string }) => item.image)
      .filter((u: string | undefined): u is string => !!u);

    if (imageUrls.length === 0) return;

    const fillGallery = (root: Element | null) => {
      if (!root) return;
      // Gradio 3/4 gallery: replace thumbnails with final result images
      let imgs = root.querySelectorAll<HTMLImageElement>('img');
      if (imgs.length === 0) {
        // Empty gallery — create a simple preview row
        const wrap = document.createElement('div');
        wrap.className = 'agent-scheduler-result-preview';
        wrap.style.display = 'flex';
        wrap.style.flexWrap = 'wrap';
        wrap.style.gap = '8px';
        for (const url of imageUrls) {
          const img = document.createElement('img');
          img.src = url;
          img.style.maxWidth = '100%';
          img.style.height = 'auto';
          wrap.appendChild(img);
        }
        root.appendChild(wrap);
        return;
      }
      imageUrls.forEach((url, i) => {
        if (imgs[i]) imgs[i].src = url;
      });
    };

    fillGallery(gradioApp().querySelector('#agent_scheduler_current_task_images'));
    if (type === 'img2img') {
      fillGallery(gradioApp().querySelector('#img2img_gallery'));
    } else {
      fillGallery(gradioApp().querySelector('#txt2img_gallery'));
    }
  } catch (e) {
    console.warn('[AgentScheduler] Failed to display task result images', e);
  }
}

function initQueueHandler() {
  const getUiCheckpoint = (is_img2img: boolean) => {
    const root = gradioApp();
    const enqueue_wrapper_model = root.querySelector<HTMLInputElement>(
      `#${is_img2img ? 'img2img_enqueue_wrapper' : 'txt2img_enqueue_wrapper'} input`
    );
    if (enqueue_wrapper_model != null) {
      const checkpoint = enqueue_wrapper_model.value;
      if (checkpoint === 'Runtime Checkpoint' || checkpoint !== 'Current Checkpoint') {
        return checkpoint;
      }
    }

    const setting_sd_model = root.querySelector<HTMLInputElement>(
      '#setting_sd_model_checkpoint input'
    );
    return setting_sd_model?.value ?? 'Current Checkpoint';
  };

  // Enqueue inputs are [checkpoint, ...generate_inputs].
  // Never use create_submit_args — it strips trailing script-arg arrays.
  // Never overwrite window.randomId here (that broke Generate after Enqueue).
  const prepareEnqueueArgs = (raw: any[], isImg2Img: boolean) => {
    const args = raw.length === 1 && Array.isArray(raw[0]) ? [...raw[0]] : [...raw];
    args[0] = getUiCheckpoint(isImg2Img);
    args[1] = makeTaskId();
    if (isImg2Img) {
      args[2] = get_tab_index('mode_img2img');
    }
    return args;
  };

  const refreshPendingSoon = () => {
    window.setTimeout(() => pendingStore.refresh(), 400);
  };

  const btnEnqueue = gradioApp().querySelector<HTMLButtonElement>('#txt2img_enqueue')!;
  window.submit_enqueue = (...args) => {
    const res = prepareEnqueueArgs(args, false);

    if (btnEnqueue != null) {
      btnEnqueue.innerText = 'Queued';
      setTimeout(() => {
        btnEnqueue.innerText = 'Enqueue';
        refreshPendingSoon();
      }, 1000);
    }

    return res;
  };

  const btnImg2ImgEnqueue = gradioApp().querySelector<HTMLButtonElement>('#img2img_enqueue')!;
  window.submit_enqueue_img2img = (...args) => {
    const res = prepareEnqueueArgs(args, true);

    if (btnImg2ImgEnqueue != null) {
      btnImg2ImgEnqueue.innerText = 'Queued';
      setTimeout(() => {
        btnImg2ImgEnqueue.innerText = 'Enqueue';
        refreshPendingSoon();
      }, 1000);
    }

    return res;
  };

  // detect queue button placement
  const interrogateCol = gradioApp().querySelector<HTMLDivElement>('.interrogate-col');
  if (interrogateCol != null && interrogateCol.childElementCount > 2) {
    interrogateCol.classList.add('has-queue-button');
  }

  // setup keyboard shortcut (settings panel may not be mounted yet)
  const setting = gradioApp().querySelector<HTMLTextAreaElement>(
    '#setting_queue_keyboard_shortcut textarea'
  );
  if (setting?.value && !setting.value.includes('Disabled')) {
    const parts = setting.value.split('+');
    const code = parts.pop();

    const handleShortcut = (e: KeyboardEvent) => {
      if (e.code !== code) return;
      if (parts.includes('Shift') && !e.shiftKey) return;
      if (parts.includes('Alt') && !e.altKey) return;
      if (parts.includes('Command') && !e.metaKey) return;
      if ((parts.includes('Control') || parts.includes('Ctrl')) && !e.ctrlKey) return;

      e.preventDefault();
      e.stopPropagation();

      const activeTab = get_tab_index('tabs');
      if (activeTab === 0) {
        btnEnqueue.click();
      } else if (activeTab === 1) {
        btnImg2ImgEnqueue.click();
      }
    };

    window.addEventListener('keydown', handleShortcut);

    const txt2imgPrompt = gradioApp().querySelector<HTMLTextAreaElement>(
      '#txt2img_prompt textarea'
    );
    txt2imgPrompt?.addEventListener('keydown', handleShortcut);

    const img2imgPrompt = gradioApp().querySelector<HTMLTextAreaElement>(
      '#img2img_prompt textarea'
    );
    img2imgPrompt?.addEventListener('keydown', handleShortcut);
  }

  // watch for current task id change
  let watchedProgressId: string | null = null;
  let pendingPollTimer: number | null = null;

  const stopPendingPoll = () => {
    if (pendingPollTimer != null) {
      window.clearInterval(pendingPollTimer);
      pendingPollTimer = null;
    }
  };

  const ensurePendingPoll = () => {
    if (pendingPollTimer != null) return;
    pendingPollTimer = window.setInterval(() => {
      const s = pendingStore.getState();
      if (s.total_pending_tasks > 0 || s.current_task_id != null) {
        pendingStore.refresh();
      } else {
        stopPendingPoll();
      }
    }, 1000);
  };

  pendingStore.subscribe((curr, prev) => {
    if (curr.total_pending_tasks > 0 || curr.current_task_id != null) {
      ensurePendingPoll();
    }

    const id = curr.current_task_id;
    if (id != null && id !== watchedProgressId) {
      watchedProgressId = id;
      const task = curr.pending_tasks.find(t => t.id === id);
      showTaskProgress(id, task?.type, () => {
        watchedProgressId = null;
        pendingStore.refresh();
        historyStore.refresh();
      });
    } else if (id == null && prev.current_task_id != null) {
      watchedProgressId = null;
    }
  });

  // Enqueue bridge notifies when a task was queued outside Gradio click wiring
  window.addEventListener('agentSchedulerQueueUpdated', () => {
    pendingStore.refresh().then(() => ensurePendingPoll());
  });

  // context menu
  const queueWithTaskName = (img2img = false) => {
    const name = prompt('Enter task name');
    const prev = window.randomId;
    window.randomId = () => name ?? makeTaskId();
    if (img2img) {
      btnImg2ImgEnqueue.click();
    } else {
      btnEnqueue.click();
    }
    window.setTimeout(() => {
      window.randomId = prev;
    }, 0);
  };
  const queueWithEveryCheckpoint = (img2img = false) => {
    const prev = window.randomId;
    window.randomId = () => '$$_queue_with_all_checkpoints_$$';
    if (img2img) {
      btnImg2ImgEnqueue.click();
    } else {
      btnEnqueue.click();
    }
    window.setTimeout(() => {
      window.randomId = prev;
    }, 0);
  };

  appendContextMenuOption('#txt2img_enqueue', 'Queue with task name', () => queueWithTaskName());
  appendContextMenuOption('#txt2img_enqueue', 'Queue with all checkpoints', () =>
    queueWithEveryCheckpoint()
  );
  appendContextMenuOption('#img2img_enqueue', 'Queue with task name', () =>
    queueWithTaskName(true)
  );
  appendContextMenuOption('#img2img_enqueue', 'Queue with all checkpoints', () =>
    queueWithEveryCheckpoint(true)
  );

  // preview modal save button
  const origModalSaveImage = window.modalSaveImage;
  window.modalSaveImage = (event: Event) => {
    const tab = gradioApp().querySelector<HTMLDivElement>('#tab_agent_scheduler');
    const visible =
      tab != null &&
      tab.style.display !== 'none' &&
      !tab.classList.contains('hidden') &&
      tab.offsetParent !== null;
    if (visible) {
      gradioApp().querySelector<HTMLButtonElement>('#agent_scheduler_save')?.click();
      event.preventDefault();
    } else {
      origModalSaveImage(event);
    }
  };
}

function initTabChangeHandler() {
  sharedStore.subscribe((curr, prev) => {
    if (!curr.uiAsTab || curr.selectedTab !== prev.selectedTab) {
      if (curr.selectedTab === 'pending') {
        pendingStore.refresh();
      } else {
        historyStore.refresh();
      }
    }
  });

  // watch for tab activation (Gradio 4 toggles .hidden / aria, not only style.display)
  const isVisible = (el: HTMLElement) => {
    if (el.classList.contains('hidden')) return false;
    if (el.style.display === 'none') return false;
    if (el.getAttribute('hidden') != null) return false;
    return true;
  };

  const observer = new MutationObserver(mutationsList => {
    mutationsList.forEach(styleChange => {
      const tab = styleChange.target as HTMLElement;
      if (!isVisible(tab)) return;

      switch (tab.id) {
        case 'tab_agent_scheduler':
          if (sharedStore.getState().selectedTab === 'pending') {
            pendingStore.refresh();
          } else {
            historyStore.refresh();
          }
          break;
        case 'agent_scheduler_pending_tasks_tab':
          sharedStore.setSelectedTab('pending');
          break;
        case 'agent_scheduler_history_tab':
          sharedStore.setSelectedTab('history');
          break;
      }
    });
  });
  const tab = gradioApp().querySelector('#tab_agent_scheduler');
  if (tab != null) {
    observer.observe(tab, { attributeFilter: ['style', 'class', 'hidden'] });
  } else {
    sharedStore.setState({ uiAsTab: false });
  }
  const pendingTab = gradioApp().querySelector('#agent_scheduler_pending_tasks_tab');
  const historyTab = gradioApp().querySelector('#agent_scheduler_history_tab');
  if (pendingTab) observer.observe(pendingTab, { attributeFilter: ['style', 'class', 'hidden'] });
  if (historyTab) observer.observe(historyTab, { attributeFilter: ['style', 'class', 'hidden'] });

  // Also refresh when user clicks the main Agent Scheduler nav button
  const navButtons = Array.from(
    gradioApp().querySelectorAll('#tabs > .tab-nav button, #tabs .tab-nav button'),
  ) as HTMLButtonElement[];
  for (const btn of navButtons) {
    if ((btn.textContent || '').includes('Agent Scheduler')) {
      btn.addEventListener('click', () => {
        window.setTimeout(() => {
          if (sharedStore.getState().selectedTab === 'pending') pendingStore.refresh();
          else historyStore.refresh();
        }, 200);
      });
    }
  }
}

function initPendingTab() {
  const store = pendingStore;

  // load data for edit
  sharedStore.getSamplers().then(res => samplers.push(...res));
  sharedStore.getCheckpoints().then(res => checkpoints.push(...res));

  // init actions
  const refreshButton = gradioApp().querySelector<HTMLButtonElement>(
    '#agent_scheduler_action_reload'
  )!;
  refreshButton.addEventListener('click', () => store.refresh());

  const pauseButton = gradioApp().querySelector<HTMLButtonElement>(
    '#agent_scheduler_action_pause'
  )!;
  pauseButton.addEventListener('click', () => store.pauseQueue().then(notify));

  const resumeButton = gradioApp().querySelector<HTMLButtonElement>(
    '#agent_scheduler_action_resume'
  )!;
  resumeButton.addEventListener('click', () => store.resumeQueue().then(notify));

  const clearButton = gradioApp().querySelector<HTMLButtonElement>(
    '#agent_scheduler_action_clear_queue'
  )!;
  clearButton.addEventListener('click', () => {
    if (confirm('Are you sure you want to clear the queue?')) {
      store.clearQueue().then(notify);
    }
  });

  const importButton = gradioApp().querySelector<HTMLButtonElement>(
    '#agent_scheduler_action_import'
  )!;
  const importInput = gradioApp().querySelector<HTMLInputElement>('#agent_scheduler_import_file')!;

  importButton.addEventListener('click', () => {
    importInput.click();
  });
  importInput.addEventListener('change', e => {
    if (e.target === null) return;

    const files = importInput.files;
    if (files == null || files.length === 0) return;

    const file = files[0];
    const reader = new FileReader();
    reader.onload = () => {
      const data = reader.result as string;
      store
        .importQueue(data)
        .then(notify)
        .then(() => {
          importInput.value = '';
          store.refresh();
        });
    };
    reader.readAsText(file);
  });

  const exportButton = gradioApp().querySelector<HTMLButtonElement>(
    '#agent_scheduler_action_export'
  )!;
  exportButton.addEventListener('click', () => {
    store.exportQueue().then(data => {
      const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(data));
      const dlAnchorElem = document.createElement('a');
      dlAnchorElem.setAttribute('href', dataStr);
      dlAnchorElem.setAttribute('download', `agent-scheduler-${Date.now()}.json`);
      dlAnchorElem.click();
    });
  });

  // watch for queue status change
  const updateUiState = (state: ReturnType<typeof store.getState>) => {
    if (state.paused) {
      pauseButton.classList.add('hide', 'hidden');
      resumeButton.classList.remove('hide', 'hidden');
    } else {
      pauseButton.classList.remove('hide', 'hidden');
      resumeButton.classList.add('hide', 'hidden');
    }
  };
  store.subscribe(updateUiState);
  updateUiState(store.getState());

  let lastHighlightedRow: RowNode<Task> | null;

  let pageMoveTimeout: ReturnType<typeof setTimeout> | null;

  const PAGE_MOVE_TIMEOUT_MS = 1.5 * 1000;
  const PAGE_MOVE_Y_MARGIN = 45 / 2; // half of default (min) rowHeight

  const clearPageMoveTimeout = () => {
    if (pageMoveTimeout != null) {
      clearTimeout(pageMoveTimeout);
      pageMoveTimeout = null;
    }
  };
  const updatePageMoveTimeout = (api: GridApi<Task>, pixel: number) => {
    if (lastHighlightedRow == null) {
      clearPageMoveTimeout();
      return;
    }

    const firstRowIndexOfPage = api.paginationGetPageSize() * api.paginationGetCurrentPage();
    const lastRowIndexOfPage = Math.min(
      api.paginationGetPageSize() * (api.paginationGetCurrentPage() + 1) - 1,
      api.getDisplayedRowCount() - 1
    );

    const rowIndex = lastHighlightedRow.rowIndex!;
    if (rowIndex === firstRowIndexOfPage) {
      if (getPixelOnRow(api, lastHighlightedRow, pixel) > PAGE_MOVE_Y_MARGIN) {
        clearPageMoveTimeout();
        return;
      }
      if (pageMoveTimeout == null) {
        pageMoveTimeout = setTimeout(() => {
          if (api.paginationGetCurrentPage() > 0) {
            api.paginationGoToPreviousPage();
            highlightRow(api);
          }
          pageMoveTimeout = null;
        }, PAGE_MOVE_TIMEOUT_MS);
      }
    } else if (rowIndex === lastRowIndexOfPage) {
      if (
        getPixelOnRow(api, lastHighlightedRow, pixel) <
        lastHighlightedRow.rowHeight! - PAGE_MOVE_Y_MARGIN
      ) {
        clearPageMoveTimeout();
        return;
      }
      if (pageMoveTimeout == null) {
        pageMoveTimeout = setTimeout(() => {
          if (api.paginationGetCurrentPage() < api.paginationGetTotalPages() - 1) {
            api.paginationGoToNextPage();
            highlightRow(api);
          }
          pageMoveTimeout = null;
        }, PAGE_MOVE_TIMEOUT_MS);
      }
    }
  };

  let lastPixel: number | null;

  const clearHighlightedRow = () => {
    clearPageMoveTimeout();
    lastPixel = null;
    if (lastHighlightedRow != null) {
      lastHighlightedRow.setHighlighted(null);
      lastHighlightedRow = null;
    }
  };
  const highlightRow = (api: GridApi<Task>, pixel?: number) => {
    if (pixel == null) {
      if (lastPixel == null) return;
      pixel = lastPixel;
    } else {
      lastPixel = pixel;
    }

    const rowNode = getRowNodeAtPixel(api, pixel) as RowNode<Task> | undefined;
    if (rowNode == null) return;

    const highlight = getHighlightPosition(api, rowNode, pixel);
    if (lastHighlightedRow != null && rowNode.id !== lastHighlightedRow.id) {
      clearHighlightedRow();
    }
    rowNode.setHighlighted(highlight);
    lastHighlightedRow = rowNode;
    updatePageMoveTimeout(api, pixel);
  };

  // init grid
  const gridOptions: GridOptions<Task> = {
    ...sharedGridOptions,
    editType: 'fullRow',
    defaultColDef: {
      ...sharedGridOptions.defaultColDef,
      editable: ({ data }) => data?.status === 'pending',
      cellDataType: false,
    },
    // each entry here represents one column
    columnDefs: [
      {
        field: 'priority',
        hide: true,
        sort: 'asc',
      },
      ...sharedGridOptions.columnDefs!,
      {
        headerName: 'Action',
        pinned: 'right',
        minWidth: 110,
        maxWidth: 110,
        resizable: false,
        editable: false,
        valueGetter: ({ data }) => data?.id,
        cellClass: 'pending-actions',
        cellRenderer: ({ api, value, data }: ICellRendererParams<Task, string>) => {
          if (data == null || value == null) return;

          const node = document.createElement('div');
          node.innerHTML = `
          <div class="inline-flex mt-1 edit-actions" role="group">
            <button type="button" title="Save" class="ts-btn-action primary ts-btn-save">
              ${saveIcon}
            </button>
            <button type="button" title="Cancel" class="ts-btn-action secondary ts-btn-cancel">
              ${cancelIcon}
            </button>
          </div>
          <div class="inline-flex mt-1 control-actions" role="group">
            <button type="button" title="Run" class="ts-btn-action primary ts-btn-run"
              ${data.status === 'running' ? 'disabled' : ''}>
              ${playIcon}
            </button>
            <button type="button" title="${data.status === 'pending' ? 'Delete' : 'Interrupt'}"
              class="ts-btn-action stop ts-btn-delete">
              ${data.status === 'pending' ? deleteIcon : cancelIcon}
            </button>
          </div>
          `;

          const btnSave = node.querySelector<HTMLButtonElement>('button.ts-btn-save')!;
          btnSave.addEventListener('click', () => {
            api.showLoadingOverlay();
            pendingStore.updateTask(data.id, data).then(res => {
              notify(res);
              api.hideOverlay();
              api.stopEditing(false);
            });
          });

          const btnCancel = node.querySelector<HTMLButtonElement>('button.ts-btn-cancel')!;
          btnCancel.addEventListener('click', () => api.stopEditing(true));

          const btnRun = node.querySelector<HTMLButtonElement>('button.ts-btn-run')!;
          btnRun.addEventListener('click', () => {
            api.showLoadingOverlay();
            store.runTask(value).then(() => api.hideOverlay());
          });
          const btnDelete = node.querySelector<HTMLButtonElement>('button.ts-btn-delete')!;
          btnDelete.addEventListener('click', () => {
            api.showLoadingOverlay();
            store.deleteTask(value).then(res => {
              notify(res);
              api.applyTransaction({ remove: [data] });
              api.hideOverlay();
            });
          });

          return node;
        },
      },
    ],
    onColumnMoved: ({ api }) => {
      const colState = api.getColumnState();
      const colStateStr = JSON.stringify(colState);
      localStorage.setItem('agent_scheduler:queue_col_state', colStateStr);
    },
    onSortChanged: ({ api }) => {
      const colState = api.getColumnState();
      const colStateStr = JSON.stringify(colState);
      localStorage.setItem('agent_scheduler:queue_col_state', colStateStr);
    },
    onColumnResized: ({ api }) => {
      const colState = api.getColumnState();
      const colStateStr = JSON.stringify(colState);
      localStorage.setItem('agent_scheduler:queue_col_state', colStateStr);
    },
    onGridReady: ({ api }) => {
      // init quick search input
      const searchInput = initSearchInput('#agent_scheduler_action_search');
      searchInput.addEventListener(
        'keyup',
        debounce(function () {
          api.updateGridOptions({ quickFilterText: this.value });
        }, 200)
      );

      const updateRowData = (state: ReturnType<typeof store.getState>) => {
        api.updateGridOptions({ rowData: state.pending_tasks });

        if (state.current_task_id != null) {
          const node = api.getRowNode(state.current_task_id);
          if (node != null) {
            api.refreshCells({ rowNodes: [node], force: true });
          }
        }

        api.clearFocusedCell();
        api.autoSizeAllColumns();
      };
      store.subscribe(updateRowData);
      updateRowData(store.getState());

      // restore col state
      const colStateStr = localStorage.getItem('agent_scheduler:queue_col_state');
      if (colStateStr != null) {
        const colState = JSON.parse(colStateStr);
        api.applyColumnState({ state: colState, applyOrder: true });
      }
    },
    onRowDragEnter: ({ api, y }) => highlightRow(api, y),
    onRowDragMove: ({ api, y }) => highlightRow(api, y),
    onRowDragLeave: () => clearHighlightedRow(),
    onRowDragEnd: ({ api, node }) => {
      const highlightedRow = lastHighlightedRow;
      if (highlightedRow == null) {
        clearHighlightedRow();
        return;
      }

      const id = node.data?.id;
      const highlightedId = highlightedRow.data?.id;
      if (id == null || highlightedId == null || id === highlightedId) {
        clearHighlightedRow();
        return;
      }

      let index = -1,
        overIndex = -1;
      const tasks = [...store.getState().pending_tasks].sort((a, b) => a.priority - b.priority);
      for (let i = 0; i < tasks.length; i++) {
        if (tasks[i].id === id) {
          index = i;
        }
        if (tasks[i].id === highlightedId) {
          overIndex = i;
        }
        if (index !== -1 && overIndex !== -1) {
          break;
        }
      }
      if (index === -1 || overIndex === -1) {
        clearHighlightedRow();
        return;
      }
      if (highlightedRow.highlighted === RowHighlightPosition.Below) {
        overIndex += 1;
      }
      if (overIndex === index || overIndex === index + 1) {
        clearHighlightedRow();
        return;
      }

      const overId = tasks[overIndex]?.id ?? 'bottom';

      api.showLoadingOverlay();
      store.moveTask(id, overId).then(() => {
        clearHighlightedRow();
        api.hideOverlay();
      });
    },
    onRowEditingStarted: ({ api, data, node }) => {
      if (data == null) return;

      node.setDataValue('editing', true);
      api.refreshCells({ rowNodes: [node], force: true });
    },
    onRowEditingStopped: ({ api, data, node }) => {
      if (data == null) return;

      node.setDataValue('editing', false);
      api.refreshCells({ rowNodes: [node], force: true });
    },
    onRowValueChanged: ({ api, data }) => {
      if (data == null) return;

      api.showLoadingOverlay();
      pendingStore.updateTask(data.id, data).then(res => {
        notify(res);
        api.hideOverlay();
      });
    },
  };

  const eGridDiv = gradioApp().querySelector<HTMLDivElement>(
    '#agent_scheduler_pending_tasks_grid'
  )!;

  if (typeof eGridDiv.dataset.pageSize === 'string') {
    const pageSize = parseInt(eGridDiv.dataset.pageSize, 10);

    if (pageSize > 0) {
      gridOptions.paginationAutoPageSize = false;
      gridOptions.paginationPageSize = pageSize;
    }
  }

  createGrid(eGridDiv, gridOptions);
}

function initHistoryTab() {
  const store = historyStore;

  // init actions
  const refreshButton = gradioApp().querySelector<HTMLButtonElement>(
    '#agent_scheduler_action_refresh_history'
  )!;
  refreshButton.addEventListener('click', () => store.refresh());
  const clearButton = gradioApp().querySelector<HTMLButtonElement>(
    '#agent_scheduler_action_clear_history'
  )!;
  clearButton.addEventListener('click', () => {
    if (!confirm('Are you sure you want to clear the history?')) return;
    store.clearHistory().then(notify);
  });
  const requeueButton = gradioApp().querySelector<HTMLButtonElement>(
    '#agent_scheduler_action_requeue'
  )!;
  requeueButton.addEventListener('click', () => {
    store.requeueFailedTasks().then(notify);
  });

  const resultTaskId = gradioApp().querySelector<HTMLTextAreaElement>(
    '#agent_scheduler_history_selected_task textarea'
  )!;
  const resultImageId = gradioApp().querySelector<HTMLTextAreaElement>(
    '#agent_scheduler_history_selected_image textarea'
  )!;
  const resultGallery = gradioApp().querySelector<HTMLDivElement>(
    '#agent_scheduler_history_gallery'
  )!;
  resultGallery.addEventListener('click', e => {
    const target = e.target as Element | null;
    if (target?.tagName === 'IMG') {
      const imageIdx = Array.prototype.indexOf.call(
        target.parentElement!.parentElement!.children,
        target.parentElement!
      );
      resultImageId.value = imageIdx.toString();
      resultImageId.dispatchEvent(new Event('input', { bubbles: true }));
    }
  });

  window.agent_scheduler_status_filter_changed = value => {
    store.onFilterStatus(value?.toLowerCase() as TaskStatus | undefined);
  };

  // init grid
  const gridOptions: GridOptions<Task> = {
    ...sharedGridOptions,
    readOnlyEdit: true,
    defaultColDef: {
      ...sharedGridOptions.defaultColDef,
      sortable: true,
      editable: ({ colDef }) => colDef?.field === 'name',
    },
    // each entry here represents one column
    columnDefs: [
      {
        headerName: '',
        field: 'bookmarked',
        minWidth: 55,
        maxWidth: 55,
        pinned: 'left',
        sort: 'desc',
        tooltipValueGetter: ({ value }: ITooltipParams<Task, boolean | undefined, any>) =>
          value === true ? 'Unbookmark' : 'Bookmark',
        cellClass: ({ value }: CellClassParams<Task, boolean | undefined>) => [
          'cursor-pointer',
          'pt-3',
          value === true ? 'ts-bookmarked' : 'ts-bookmark',
        ],
        cellRenderer: ({ value }: ICellRendererParams<Task, boolean | undefined>) =>
          value === true ? bookmarked : bookmark,
        onCellClicked: ({
          api,
          data,
          value,
          event,
        }: CellClickedEvent<Task, boolean | undefined>) => {
          if (data == null) return;

          if (event != null) {
            event.stopPropagation();
            event.preventDefault();
          }

          const bookmarked = value === true;
          store.bookmarkTask(data.id, !bookmarked).then(res => {
            notify(res);
            api.applyTransaction({
              update: [{ ...data, bookmarked: !bookmarked }],
            });
          });
        },
      },
      {
        field: 'priority',
        hide: true,
        sort: 'desc',
      },
      {
        ...sharedGridOptions.columnDefs![0],
        rowDrag: false,
      },
      ...sharedGridOptions.columnDefs!.slice(1),
      {
        headerName: 'Action',
        pinned: 'right',
        minWidth: 110,
        maxWidth: 110,
        resizable: false,
        valueGetter: ({ data }) => data?.id,
        cellRenderer: ({ api, data, value }: ICellRendererParams<Task, string | undefined>) => {
          if (data == null || value == null) return;

          const node = document.createElement('div');
          node.innerHTML = `
          <div class="inline-flex mt-1" role="group">
            <button type="button" title="Requeue" class="ts-btn-action primary ts-btn-run">
              ${rotateIcon}
            </button>
            <button type="button" title="Delete" class="ts-btn-action stop ts-btn-delete">
              ${deleteIcon}
            </button>
          </div>
          `;

          const btnRun = node.querySelector<HTMLButtonElement>('button.ts-btn-run')!;
          btnRun.addEventListener('click', e => {
            e.preventDefault();
            e.stopPropagation();
            store.requeueTask(value).then(notify);
          });
          const btnDelete = node.querySelector<HTMLButtonElement>('button.ts-btn-delete')!;
          btnDelete.addEventListener('click', e => {
            e.preventDefault();
            e.stopPropagation();
            api.showLoadingOverlay();
            pendingStore.deleteTask(value).then(res => {
              notify(res);
              api.applyTransaction({ remove: [data] });
              api.hideOverlay();
            });
          });

          return node;
        },
      },
    ],
    rowSelection: 'single',
    suppressRowDeselection: true,
    onColumnMoved: ({ api }) => {
      const colState = api.getColumnState();
      const colStateStr = JSON.stringify(colState);
      localStorage.setItem('agent_scheduler:history_col_state', colStateStr);
    },
    onSortChanged: ({ api }) => {
      const colState = api.getColumnState();
      const colStateStr = JSON.stringify(colState);
      localStorage.setItem('agent_scheduler:history_col_state', colStateStr);
    },
    onColumnResized: ({ api }) => {
      const colState = api.getColumnState();
      const colStateStr = JSON.stringify(colState);
      localStorage.setItem('agent_scheduler:history_col_state', colStateStr);
    },
    onGridReady: ({ api }) => {
      // init quick search input
      const searchInput = initSearchInput('#agent_scheduler_action_search_history');
      searchInput.addEventListener(
        'keyup',
        debounce(function () {
          api.updateGridOptions({ quickFilterText: this.value });
        }, 200)
      );

      const updateRowData = (state: ReturnType<typeof store.getState>) => {
        api.updateGridOptions({ rowData: state.tasks });
        api.clearFocusedCell();
        api.autoSizeAllColumns();
      };
      store.subscribe(updateRowData);
      updateRowData(store.getState());

      // restore col state
      const colStateStr = localStorage.getItem('agent_scheduler:history_col_state');
      if (colStateStr != null) {
        const colState = JSON.parse(colStateStr);
        api.applyColumnState({ state: colState, applyOrder: true });
      }
    },
    onSelectionChanged: ({ api }) => {
      const [selected] = api.getSelectedRows();
      resultTaskId.value = selected.id;
      resultTaskId.dispatchEvent(new Event('input', { bubbles: true }));
    },
    onCellEditRequest: ({ api, data, colDef, newValue }) => {
      if (colDef.field !== 'name') return;

      const name = newValue as string | undefined;
      if (name == null) return;

      api.showLoadingOverlay();
      historyStore.renameTask(data.id, name).then(res => {
        notify(res);
        const newData = { ...data, name };
        api.applyTransaction({ update: [newData] });
        api.hideOverlay();
      });
    },
  };
  const eGridDiv = gradioApp().querySelector<HTMLDivElement>(
    '#agent_scheduler_history_tasks_grid'
  )!;

  if (typeof eGridDiv.dataset.pageSize === 'string') {
    const pageSize = parseInt(eGridDiv.dataset.pageSize, 10);

    if (pageSize > 0) {
      gridOptions.paginationAutoPageSize = false;
      gridOptions.paginationPageSize = pageSize;
    }
  }

  createGrid(eGridDiv, gridOptions);
}

let queueHandlerReady = false;
let agentSchedulerInitialized = false;

/**
 * Lobe SplitView can leave #*_enqueue visually clickable while Gradio's
 * Svelte click handler no longer fires. Bridge: click Enqueue → click Generate
 * (collect inputs) → rewrite queue/predict fn_index to enqueue.
 * Plain JS twin: javascript/agent-scheduler.enqueue-bridge.js
 * (must load before agent-scheduler.iife.js — filename sorts first —
 *  and claims __agentSchedulerEnqueueBridge so the stale rewrite bridge is skipped).
 */
function installEnqueueBridge() {
  const w = window as Window & {
    __agentSchedulerEnqueueBridge?: boolean;
    __agentSchedulerEnqueueActive?: boolean;
    __agentSchedulerSubmitPatched?: boolean;
    gradio_config?: {
      components: { id: number; props?: { elem_id?: string } }[];
      dependencies: { targets?: [number, string][]; inputs?: unknown[] }[];
    };
  };
  if (w.__agentSchedulerEnqueueBridge) return;
  // Prefer the queue/ui plain-JS bridge when it already claimed the flag
  if ((w as any).__agentSchedulerEnqueueBridgeUi) return;
  w.__agentSchedulerEnqueueBridge = true;

  type Pending = { enqueueFn: number; isImg2Img: boolean; genFns: number[] };
  let pending: Pending | null = null;

  const root = () => gradioApp();

  const getCheckpoint = (isImg2Img: boolean) => {
    const wrap = root().querySelector<HTMLInputElement>(
      `#${isImg2Img ? 'img2img' : 'txt2img'}_enqueue_wrapper input`
    );
    if (wrap?.value && wrap.value !== 'Current Checkpoint') return wrap.value;
    const setting = root().querySelector<HTMLInputElement>(
      '#setting_sd_model_checkpoint input'
    );
    return setting?.value ?? 'Current Checkpoint';
  };

  const findComponentId = (elemId: string) => {
    const cfg = w.gradio_config;
    if (!cfg) return null;
    return cfg.components.find((c) => c.props?.elem_id === elemId)?.id ?? null;
  };

  const findClickFnIndexes = (elemId: string) => {
    const cfg = w.gradio_config;
    const compId = findComponentId(elemId);
    if (compId == null || !cfg) return [] as number[];
    return cfg.dependencies
      .map((d, i) => ({
        i,
        n: (d.inputs || []).length,
        hit: (d.targets || []).some((t) => t[0] === compId && t[1] === 'click'),
      }))
      .filter((x) => x.hit)
      .sort((a, b) => b.n - a.n)
      .map((x) => x.i);
  };

  const rewriteBody = (url: string, bodyStr: string) => {
    if (!pending) return null;
    if (!/\/(queue\/join|run\/predict|call\/)/.test(url)) return null;
    try {
      const body = JSON.parse(bodyStr) as {
        fn_index?: number;
        data?: unknown[];
        trigger_id?: number;
      };
      if (typeof body.fn_index !== 'number') return null;
      if (!pending.genFns.includes(body.fn_index)) return null;
      body.fn_index = pending.enqueueFn;
      if (Array.isArray(body.data)) body.data = [getCheckpoint(pending.isImg2Img), ...body.data];
      const enqId = findComponentId(pending.isImg2Img ? 'img2img_enqueue' : 'txt2img_enqueue');
      if (enqId != null) body.trigger_id = enqId;
      console.info(
        `[AgentScheduler] bridge: generate→enqueue fn=${pending.enqueueFn} n=${body.data?.length}`
      );
      pending = null;
      w.__agentSchedulerEnqueueActive = false;
      return JSON.stringify(body);
    } catch (e) {
      console.error('[AgentScheduler] bridge rewrite failed', e);
      return null;
    }
  };

  const origFetch = window.fetch.bind(window);
  window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
    const url =
      typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    if (pending && init && typeof init.body === 'string') {
      const rewritten = rewriteBody(url, init.body);
      if (rewritten != null) init = { ...init, body: rewritten };
    }
    return origFetch(input, init);
  };

  if (!w.__agentSchedulerSubmitPatched) {
    w.__agentSchedulerSubmitPatched = true;
    (['submit', 'submit_img2img'] as const).forEach((name) => {
      const orig = w[name];
      if (typeof orig !== 'function') return;
      (w as any)[name] = function (...args: any[]) {
        if (!w.__agentSchedulerEnqueueActive) return orig.apply(this, args);
        const id = typeof randomId === 'function' ? randomId() : `task(${Math.random()})`;
        let res = args.length === 1 && Array.isArray(args[0]) ? [...args[0]] : [...args];
        res[0] = id;
        if (name === 'submit_img2img' && typeof get_tab_index === 'function') {
          res[1] = get_tab_index('mode_img2img');
        }
        return res;
      };
    });
  }

  const startBridge = (isImg2Img: boolean, ev: Event) => {
    ev.preventDefault();
    ev.stopPropagation();
    ev.stopImmediatePropagation();

    const enqFns = findClickFnIndexes(isImg2Img ? 'img2img_enqueue' : 'txt2img_enqueue');
    const genFns = findClickFnIndexes(isImg2Img ? 'img2img_generate' : 'txt2img_generate');
    const genBtn = root().querySelector<HTMLButtonElement>(
      isImg2Img ? '#img2img_generate' : '#txt2img_generate'
    );
    if (!enqFns.length || !genFns.length || !genBtn) {
      console.error('[AgentScheduler] bridge: missing deps', { enqFns, genFns, genBtn });
      return;
    }

    pending = { enqueueFn: enqFns[0], isImg2Img, genFns };
    w.__agentSchedulerEnqueueActive = true;
    const btn = (ev.target as HTMLElement | null)?.closest?.('button');
    if (btn) {
      const prev = btn.innerText;
      btn.innerText = 'Queued';
      setTimeout(() => {
        btn.innerText = prev || 'Enqueue';
      }, 1200);
    }
    window.setTimeout(() => {
      if (pending) {
        console.warn('[AgentScheduler] bridge: timed out');
        pending = null;
        w.__agentSchedulerEnqueueActive = false;
      }
    }, 8000);
    genBtn.click();
  };

  // Listen on gradioApp() (shadow root). document retargets target → <gradio-app>.
  gradioApp().addEventListener(
    'click',
    (ev) => {
      const t = ev.target as HTMLElement | null;
      if (!t?.closest) return;
      if (t.closest('#txt2img_enqueue')) startBridge(false, ev);
      else if (t.closest('#img2img_enqueue')) startBridge(true, ev);
    },
    true
  );
  console.info('[AgentScheduler] Enqueue bridge installed on gradioApp()');
}

onUiLoaded(function initEnqueueHandlers() {
  // Enqueue must be ready ASAP — do not wait for Agent Scheduler tab grids
  if (gradioApp().querySelector('#txt2img_enqueue') == null) {
    setTimeout(initEnqueueHandlers, 300);
    return;
  }
  if (queueHandlerReady) return;
  try {
    installEnqueueBridge();
    initQueueHandler();
    queueHandlerReady = true;
    console.info('[AgentScheduler] Enqueue handlers ready');
  } catch (error) {
    console.error('[AgentScheduler] Enqueue init failed', error);
    setTimeout(initEnqueueHandlers, 800);
  }
});

onUiLoaded(function initAgentScheduler() {
  if (gradioApp().querySelector('#agent_scheduler_tabs') == null) {
    setTimeout(initAgentScheduler, 500);
    return;
  }

  if (agentSchedulerInitialized) return;

  try {
    if (!queueHandlerReady) {
      initQueueHandler();
      queueHandlerReady = true;
    }
    initTabChangeHandler();
    initPendingTab();
    initHistoryTab();
    agentSchedulerInitialized = true;
    console.info('[AgentScheduler] UI initialized');
  } catch (error) {
    console.error('[AgentScheduler] UI init failed', error);
    window.setTimeout(() => {
      if (!agentSchedulerInitialized) initAgentScheduler();
    }, 1500);
  }
});
