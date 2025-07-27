"use client"

interface TrendingBarProps {
  onTrendingClick: (query: string) => void
}

const trendingItems = [
  "iPhone 15",
  "Samsung Galaxy",
  "Kurta",
  "AirPods",
  "Laptop",
  "Shoes",
  "Watch",
  "Headphones",
  "Mobile Cover",
  "Bluetooth Speaker",
]

export default function TrendingBar({ onTrendingClick }: TrendingBarProps) {
  return (
    <div className="bg-white border-b shadow-sm">
      <div className="px-4 py-3">
        <div className="flex items-center space-x-3">
          <span className="text-xs md:text-sm font-semibold text-gray-700 whitespace-nowrap">🔥 Trending:</span>
          <div className="flex space-x-2 md:space-x-3 overflow-x-auto scrollbar-hide">
            {trendingItems.map((item) => (
              <button
                key={item}
                onClick={() => onTrendingClick(item)}
                className="bg-gradient-to-r from-blue-50 to-blue-100 hover:from-blue-100 hover:to-blue-200 px-2 md:px-4 py-1 md:py-2 rounded-full text-xs md:text-sm font-medium whitespace-nowrap transition-all duration-200 border border-blue-200 hover:border-blue-300"
              >
                {item}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
