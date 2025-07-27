"use client"

interface Filters {
  categories: string[]
  brands: string[]
  rating: number
  priceMin: number
  priceMax: number
  deliveryDays: number
}

interface FiltersSidebarProps {
  filters: Filters
  onChange: (filters: Filters) => void
}

const categories = ["Electronics", "Fashion", "Home & Kitchen", "Books", "Sports"]

const brands = ["Apple", "Samsung", "Nike", "Adidas", "Sony", "LG", "Boat", "Realme"]

export default function FiltersSidebar({ filters, onChange }: FiltersSidebarProps) {
  const handleCategoryChange = (category: string, checked: boolean) => {
    const newCategories = checked ? [...filters.categories, category] : filters.categories.filter((c) => c !== category)

    onChange({ ...filters, categories: newCategories })
  }

  const handleBrandChange = (brand: string, checked: boolean) => {
    const newBrands = checked ? [...filters.brands, brand] : filters.brands.filter((b) => b !== brand)

    onChange({ ...filters, brands: newBrands })
  }

  return (
    <div className="w-full lg:w-64 bg-white border-r lg:min-h-screen p-4">
      <h3 className="font-semibold text-lg mb-4 lg:block hidden">Filters</h3>

      {/* Mobile: Show filters in grid, Desktop: Show in column */}
      <div className="space-y-4 lg:space-y-6">
        {/* Categories */}
        <div className="lg:mb-6">
          <h4 className="font-medium mb-3 text-sm lg:text-base">Categories</h4>
          <div className="grid grid-cols-2 lg:grid-cols-1 gap-2">
            {categories.map((category) => (
              <label key={category} className="flex items-center text-sm">
                <input
                  type="checkbox"
                  checked={filters.categories.includes(category)}
                  onChange={(e) => handleCategoryChange(category, e.target.checked)}
                  className="mr-2"
                />
                <span className="text-xs lg:text-sm">{category}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Brands */}
        <div className="lg:mb-6">
          <h4 className="font-medium mb-3 text-sm lg:text-base">Brand</h4>
          <div className="grid grid-cols-2 lg:grid-cols-1 gap-2">
            {brands.map((brand) => (
              <label key={brand} className="flex items-center text-sm">
                <input
                  type="checkbox"
                  checked={filters.brands.includes(brand)}
                  onChange={(e) => handleBrandChange(brand, e.target.checked)}
                  className="mr-2"
                />
                <span className="text-xs lg:text-sm">{brand}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Rating */}
        <div className="lg:mb-6">
          <h4 className="font-medium mb-3 text-sm lg:text-base">Customer Rating</h4>
          <div className="grid grid-cols-2 lg:grid-cols-1 gap-2">
            {[4, 3, 2, 1].map((rating) => (
              <label key={rating} className="flex items-center text-sm">
                <input
                  type="radio"
                  name="rating"
                  checked={filters.rating === rating}
                  onChange={() => onChange({ ...filters, rating })}
                  className="mr-2"
                />
                <span className="text-xs lg:text-sm">{rating}★ & up</span>
              </label>
            ))}
          </div>
        </div>

        {/* Price Range */}
        <div className="lg:mb-6">
          <h4 className="font-medium mb-3 text-sm lg:text-base">Price</h4>
          <div className="flex space-x-2 mb-2">
            <input
              type="number"
              placeholder="Min"
              value={filters.priceMin || ""}
              onChange={(e) => onChange({ ...filters, priceMin: Number(e.target.value) || 0 })}
              className="w-full lg:w-20 px-2 py-1 border rounded text-sm"
            />
            <span className="text-sm self-center">to</span>
            <input
              type="number"
              placeholder="Max"
              value={filters.priceMax || ""}
              onChange={(e) => onChange({ ...filters, priceMax: Number(e.target.value) || 50000 })}
              className="w-full lg:w-20 px-2 py-1 border rounded text-sm"
            />
          </div>
        </div>

        {/* Delivery Days */}
        <div className="lg:mb-6">
          <h4 className="font-medium mb-3 text-sm lg:text-base">Delivery</h4>
          <div className="mb-2">
            <label className="text-xs lg:text-sm">Within {filters.deliveryDays} days</label>
            <input
              type="range"
              min="1"
              max="7"
              value={filters.deliveryDays}
              onChange={(e) => onChange({ ...filters, deliveryDays: Number(e.target.value) })}
              className="w-full mt-1"
            />
          </div>
        </div>
      </div>
    </div>
  )
}
