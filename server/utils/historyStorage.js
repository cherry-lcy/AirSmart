const MAX_RECORDS = 100;
let controlHistory = [];

const addRecord = (record) => {
  controlHistory.unshift({
    ...record,
    key:String(controlHistory.length+1)
  });
  if (controlHistory.length > MAX_RECORDS) {
    controlHistory.pop();
  }
};

const getHistory = () => {
  return {
    data:[...controlHistory],
    total_count:controlHistory.length
  };
};

const clearAllHistory = () => {
    controlHistory = []
}

module.exports = { addRecord, getHistory, clearAllHistory };