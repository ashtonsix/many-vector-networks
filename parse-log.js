const fs = require('fs')

const log = fs.readFileSync('selu.log', 'utf8')

const nets = log.match(/[A-Z]Net.*\n/gm)
const results = log.split(/[A-Z]Net.*\n/gm).slice(1)

const maxBy = (arr, key) => {
  if (typeof key !== 'function') {
    const _key = key
    key = v => v[_key]
  }
  let val = arr[0]
  let max = -Infinity
  for (const i in arr)
    if (key(arr[i]) > max) {
      max = key(arr[i])
      val = arr[i]
    }
  return val
}

let parsed = {}

for (let i = 0; i < nets.length; i++) {
  const net = nets[i].trim()
  const result = results[i]
  let [rows, cols, labels, title] = result
    .match(/(\d+) rows, (\d+) cols, (\d+) unique labels in '([\w\-]+)'/)
    .slice(1)
    .map(s => s.trim())
  rows = parseInt(rows, 10)
  cols = parseInt(cols, 10)
  labels = parseInt(labels, 10)

  const [loss, accuracy] = maxBy(
    result
      .split('lr=')
      .slice(1)
      .map(s => {
        return s
          .trim()
          .split(/\s+/)
          .slice(-2)
          .map(v => parseFloat(v))
      }),
    0
  )

  if (!parsed[title]) parsed[title] = {}
  parsed[title][net] = {title, net, loss, accuracy, rows, cols, labels}
}

parsed = Object.values(parsed)
parsed.sort((a, b) => (a['RNet'].title > b['RNet'].title ? 1 : -1))

console.log(',Rows,Labels,RNet,SNet,MNet - 1block, MNet - 3block')
parsed.forEach(v => {
  const title = v.RNet.title
  const r_ = v['RNet'].rows
  const l_ = v['RNet'].labels
  const r = (v['RNet'].accuracy * 100).toFixed(1)
  const s = (v['SNet'].accuracy * 100).toFixed(1)
  const m1 = (v['MNet - 1block'].accuracy * 100).toFixed(1)
  const m3 = (v['MNet - 3block'].accuracy * 100).toFixed(1)
  console.log(`${title},${r_},${l_},${r},${s},${m1},${m3}`)
})
